
import math
import lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from lib.utils import load_torch_file
from omegaconf import OmegaConf 
from pathlib import Path
from diffusers import DDPMScheduler
from lib.sgm import GeneralConditioner
from torch_ema import ExponentialMovingAverage
from lib.wrappers import AutoencoderKLWrapper, UnetWrapper
from lib.utils import rank_zero_print
from lightning.pytorch.utilities import rank_zero_only

        
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def min_snr_weighted_loss(eps_pred:torch.Tensor, eps:torch.Tensor, timesteps, noise_scheduler, snr_gamma):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2

    mse_loss_weights = (
        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    )
    # We first calculate the original loss. Then we mean over the non-batch dimensions and
    # rebalance the sample-wise losses with their respective loss weights.
    # Finally, we take the mean of the rebalanced loss.
    loss = F.mse_loss(eps_pred.float(), eps.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    loss = loss.mean()
    return loss


# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()
        
    def init_model(self):
        config = self.config
        rank_zero_print(f"Loading model from {self.model_path}")
        sd = load_torch_file(self.model_path)
        
        key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"

        config_file = Path(self.model_path).with_suffix(".yaml")
        if not config_file.is_file():
            # model_type = "v1"
            sd_legacy = True
            config_file = "lib/model_configs/sd_1.yaml"
            if key_name_v2_1 in sd and sd[key_name_v2_1].shape[-1] == 1024:
                # model_type = "v2"
                config_file = "lib/model_configs/sd_2_1.yaml"
            elif key_name_sd_xl_base in sd:
                sd_legacy = False
                config_file = "lib/model_configs/sd_xl_base.yaml"
            elif key_name_sd_xl_refiner in sd:
                sd_legacy = False
                config_file = "lib/model_configs/sd_xl_refiner.yaml"
        
        # sd 1.x fix
        sd1x_clip_key = "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight"
        if sd1x_clip_key in sd:
            new_dict = {}
            unused_keys = []
            for key in sd.keys():
                if "cond_stage_model.transformer.text_model" in key:
                    newkey = key.replace("cond_stage_model.transformer", "cond_stage_model.embedders.0.transformer")
                    unused_keys.append(key)
                    new_dict[newkey] = sd[key]
                elif "model_ema" in key:
                    unused_keys.append(key)
            sd.update(new_dict)
            for key in unused_keys:
                del sd[key]
            
        self.model_config = OmegaConf.load(config_file)
        model_params = self.model_config.model.params
        
        for item in model_params.conditioner_config.params.emb_models:
            item["target"] = item["target"].replace("modules.", "")
            item["target"] = "lib." + item["target"]   
        
        if self.config.trainer.use_xformers:
            model_params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
            model_params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers"
        
        encoder = AutoencoderKLWrapper(**model_params.first_stage_config.params).eval()
        encoder.train = disabled_train
        encoder.requires_grad_(False)
        for param in encoder.parameters():
            param.requires_grad = False
            
        self.first_stage_model = encoder
        self.scale_factor = model_params.scale_factor
        
        self.model = UnetWrapper(model_params.network_config.params)
        for conditioner in model_params.conditioner_config.params.emb_models:
            if "CLIPEmbedder" not in conditioner.target:
                continue
            conditioner.params["max_length"] = self.config.dataset.get("max_token_length", 75) + 2
        
        conditioner = GeneralConditioner(**model_params.conditioner_config.params)
        if sd_legacy:
            self.cond_stage_model = conditioner
        else:
            self.conditioner = conditioner
            
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
            
        self.to(self.target_device)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            rank_zero_print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            rank_zero_print(f"Unexpected Keys: {unexpected}")
            
        try:
            torch.compile(self.model, mode="max-autotune", fullgraph=True, dynamic=True)
        except Exception as e:
            rank_zero_print(f"Failed to compile model: {e}")
            
        self.cast_dtype = torch.float32
        self.get_conditioner = lambda: self.conditioner if hasattr(self, "conditioner") else self.cond_stage_model
        
        if config.trainer.use_ema: 
            self.model_ema = ExponentialMovingAverage(self.model.parameters(), decay=0.9999)
            rank_zero_print(f"EMA is enabled with decay {self.model_ema.decay}")
            
        # self.use_latent_cache = self.config.dataset.get("cache_latents")

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=False):
            z = self.first_stage_model.encode(x).sample()
        z = self.scale_factor * z
        return z

    @torch.inference_mode()
    @rank_zero_only 
    def sample(self, prompt, negative_prompt, generator=None, size=(1024,1024), steps=20, guidance_scale=7.5):
        self.model.eval()
        self.conditioner.cuda()
        # first construct batch
        prompts_batch = {
            "target_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "original_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "crop_coords_top_left": torch.stack([torch.asarray((0,0))]).cuda(),
        }
        prompts_batch["prompts"] = prompt
        cond = self.conditioner(prompts_batch)
        prompts_batch["prompts"] = negative_prompt
        uncond = self.conditioner(prompts_batch)
        cond = {
            "crossattn": torch.cat([uncond["crossattn"], cond["crossattn"]], dim=0).cuda().float(),
            "vector": torch.cat([uncond["vector"], cond["vector"]], dim=0).cuda().float(),
        }
        
        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = latents * self.noise_scheduler.init_noise_sigma
        
        self.noise_scheduler.set_timesteps(steps)
        timesteps = self.noise_scheduler.timesteps
        num_latent_input = 2
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.cuda().to(torch.float32)

            noise_pred = self.model(latent_model_input, torch.asarray([t]).cuda(), cond)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents.cuda()).prev_sample

        self.first_stage_model.cuda()
        image = torch.clamp((self.decode_first_stage(latents) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
    
        self.model.train()
        return image

    def forward(self, batch):  
        if "latents" not in batch.keys():
            latents = self.encode_first_stage(batch["images"])
            if torch.any(torch.isnan(latents)):
                rank_zero_print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                
            del batch["images"]
        else:
            self.first_stage_model.cpu()
            latents = batch["latents"] 
        
        if "conds" not in batch.keys():
            cond = self.get_conditioner()(batch)
        else:
            self.get_conditioner().cpu()
            cond = batch["conds"]
            cond = {k: v.to(self.model.dtype) for k, v in cond.items()}

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=self.model.dtype)
        if self.config.trainer.get("offset_noise"):
            noise = torch.randn_like(latents) + float(self.config.trainer.get("offset_noise_val")) \
                * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
        
        # https://arxiv.org/abs/2301.11706
        if self.config.trainer.get("input_perturbation"):
            noise = noise + float(self.config.trainer.get("input_perturbation_val")) * torch.randn_like(noise)

        bsz = latents.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
        # Predict the noise residual
        noise_pred = self.model(noisy_latents, timesteps, cond)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if not self.config.trainer.get("min_snr"):
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")  
        else:
            gamma = self.config.trainer.get("min_snr_val")
            loss = min_snr_weighted_loss(noise_pred.float(), target.float(), timesteps, self.noise_scheduler, gamma)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")
                
        return loss
