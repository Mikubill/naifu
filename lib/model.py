
import math
import lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from omegaconf import OmegaConf 
from pathlib import Path
from torch_ema import ExponentialMovingAverage

from lib.sgm import GeneralConditioner
from lib.sgm.denoiser import DiscreteSampling, DiscreteDenoiser
from lib.wrappers import AutoencoderKLWrapper, UnetWrapper
from lib.utils import load_torch_file, rank_zero_print
from lib.sgm.encoder_util import append_dims

from diffusers import EulerDiscreteScheduler
from lightning.pytorch.utilities import rank_zero_only

        
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


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
            
        self.to(self.target_device)
        rank_zero_print(f"Loading model from {self.model_path}")
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
        
        self.sigma_sampler = DiscreteSampling()
        self.denoiser = DiscreteDenoiser()
        self.offset_noise_level = self.config.trainer.get("offset_noise_val")
        self.type = loss_type = "l2"
        self.extra_config = self.config.get("extra", None)
        
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
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        
        # first construct batch
        model_dtype = next(self.model.parameters()).dtype
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
            "crossattn": torch.cat([uncond["crossattn"], cond["crossattn"]], dim=0).cuda().to(model_dtype),
            "vector": torch.cat([uncond["vector"], cond["vector"]], dim=0).cuda().to(model_dtype),
        }
        
        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma
        
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        num_latent_input = 2
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.cuda().to(model_dtype)

            noise_pred = self.model(latent_model_input, torch.asarray([t]).cuda(), cond)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents.cuda()).prev_sample

        self.first_stage_model.cuda()
        image = torch.clamp((self.decode_first_stage(latents) + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
    
        self.model.train()
        return image

    def forward(self, batch):  
        if "latents" not in batch.keys():
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["images"].to(self.first_stage_model.dtype))
            if torch.any(torch.isnan(latents)):
                rank_zero_print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                
            del batch["images"]
        else:
            self.first_stage_model.cpu()
            latents = batch["latents"] 
        
        if "conds" not in batch.keys():
            self.get_conditioner().to(self.target_device)
            cond = self.get_conditioner()(batch)
        else:
            self.get_conditioner().cpu()
            cond = batch["conds"]
        
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}
        latents = latents.to(model_dtype)

        loss = self.loss_fn(self.model, cond, latents)
        loss = loss.mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")
                
        return loss
    
    def loss_fn(self, network, cond, input):
        if self.extra_config is not None:
            step_start = self.extra_config.get("timestep_start", 0)
            step_end = self.extra_config.get("timestep_end", 1000)
            
            n_samples = input.shape[0]
            rand = torch.randint(step_start, step_end, (n_samples,)),
            sigmas = self.sigma_sampler(n_samples, rand=rand).to(input.device)
        else:
            sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
            
        noise = torch.randn_like(input, device=input.device)
        if self.offset_noise_level > 0.0 and self.config.trainer.offset_noise:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = self.denoiser(network, noised_input, sigmas, cond)
        w = append_dims(self.denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss