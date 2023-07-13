
import math
import lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from lightning.pytorch.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from lib.utils import min_snr_weighted_loss, load_torch_file
from omegaconf import OmegaConf 
from pathlib import Path
from diffusers import DDIMScheduler
from lib.sgm import GeneralConditioner
from lib.wrappers import AutoencoderKLWrapper, UnetWrapper

from PIL import Image
from tqdm.auto import tqdm
        
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.batch_size = config.trainer.batch_size 
        self.lr = self.config.optimizer.params.get("lr", 1e-4)
        self.init_model()
        
    def init_model(self):
        config = self.config
        print(f"Loading model from {self.model_path}")
        sd = load_torch_file(self.model_path)
        
        key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
        key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"

        config_file = Path(self.model_path).with_suffix(".yaml")
        if not config_file.is_file():
            # model_type = "v1"
            config_file = "lib/model_configs/sd_1.yaml"
            if key_name_v2_1 in sd and sd[key_name_v2_1].shape[-1] == 1024:
                # model_type = "v2"
                config_file = "lib/model_configs/sd_2_1.yaml"
            elif key_name_sd_xl_base in sd:
                config_file = "lib/model_configs/sd_xl_base.yaml"
            elif key_name_sd_xl_refiner in sd:
                config_file = "lib/model_configs/sd_xl_refiner.yaml"
            
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
        for param in encoder.parameters():
            param.requires_grad = False
        self.first_stage_model = encoder
        self.scale_factor = model_params.scale_factor

        self.model = UnetWrapper(model_params.network_config.params)
        self.conditioner = GeneralConditioner(**model_params.conditioner_config.params)
        self.noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
            
        self.cast_dtype = torch.float32
        self.conditioner.to(torch.float16)    
        if config.trainer.use_ema: 
            self.model_ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
            
        # self.use_latent_cache = self.config.dataset.get("cache_latents")

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model._decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=False):
            z = self.first_stage_model._encode(x)
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
        
        self.conditioner.cpu()
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
        image = self.decode_first_stage(latents)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        
        self.first_stage_model.cpu()
        self.model.train()
        return image

    def forward(self, batch):  
        if "latents" not in batch.keys():
            latents = self.encode_first_stage(batch["images"])
            if torch.any(torch.isnan(latents)):
                print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                
            del batch["images"]
            cond = self.conditioner(batch)
        else:
            latents = batch["latents"]
            cond = batch["conds"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
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
            loss = min_snr_weighted_loss(noise_pred.float(), target.float(), timesteps, self.noise_scheduler, gamma=gamma)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")
                
        return loss

