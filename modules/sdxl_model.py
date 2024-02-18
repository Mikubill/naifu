import torch
import torch.utils.checkpoint
import lightning as pl
from PIL import Image
from omegaconf import OmegaConf

import torch

from models.sgm import GeneralConditioner
from modules.sdxl_utils import disabled_train, UnetWrapper, AutoencoderKLWrapper
from modules.utils import apply_zero_terminal_snr, cache_snr_values   
from common.utils import load_torch_file, rank_zero_print, EmptyInitWrapper

from diffusers import EulerDiscreteScheduler, DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file


# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def build_models(self, init_unet=True, init_vae=True, init_conditioner=True):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})

        model_config = OmegaConf.load("modules/sdxl_base.yaml")
        model_params = model_config.model.params

        if trainer_cfg.use_xformers:
            unet_config = model_params.network_config.params
            vae_config = model_params.first_stage_config.params
            unet_config.spatial_transformer_attn_type = "softmax-xformers"
            vae_config.ddconfig.attn_type = "vanilla-xformers"

        for conditioner in model_params.conditioner_config.params.emb_models:
            if "CLIPEmbedder" not in conditioner.target:
                continue
            self.max_token_length = self.config.dataset.get("max_token_length", 75) + 2
            conditioner.params["device"] = str(self.target_device)
            conditioner.params["max_length"] = self.max_token_length

        tte_1 = advanced.get("train_text_encoder_1", False)
        tte_2 = advanced.get("train_text_encoder_2", False)
        model_params.conditioner_config.params.emb_models[0]["is_trainable"] = tte_1
        model_params.conditioner_config.params.emb_models[1]["is_trainable"] = tte_2

        self.scale_factor = model_params.scale_factor
        with EmptyInitWrapper(self.target_device):
            vae_config = model_params.first_stage_config.params
            unet_config = model_params.network_config.params
            cond_config = model_params.conditioner_config.params

            vae = AutoencoderKLWrapper(**vae_config) if init_vae else None
            unet = UnetWrapper(unet_config) if init_unet else None
            conditioner = (
                GeneralConditioner(**cond_config) if init_conditioner else None
            )

        vae.train = disabled_train
        vae.eval()
        vae.requires_grad_(False)
        return vae, unet, conditioner

    def init_model(self):
        advanced = self.config.get("advanced", {})
        sd = load_torch_file(self.model_path, self.target_device)
        self.first_stage_model, self.model, self.conditioner = self.build_models()
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        self.to(self.target_device)

        rank_zero_print(f"Loading model from {self.model_path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            rank_zero_print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            rank_zero_print(f"Unexpected Keys: {unexpected}")
            
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size
            
        if advanced.zero_terminal_snr:
            apply_zero_terminal_snr(self.noise_scheduler)
        cache_snr_values(self.noise_scheduler, self.target_device)
            
    def encode_pixels(self, inputs):
        feed_pixel_values = inputs
        latents = []
        for i in range(0, feed_pixel_values.shape[0], self.vae_encode_bsz):
            latents.append(
                self.vae.encode(feed_pixel_values[i : i + self.vae_encode_bsz]).latent_dist.sample()
            )
        latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad() 
    def encode_first_stage(self, x):
        latents = []
        with torch.autocast("cuda", enabled=False):
            for i in range(0, x.shape[0], self.vae_encode_bsz):
                latents.append(self.first_stage_model.encode(x).sample())
        z = torch.cat(latents, dim=0)
        z = self.scale_factor * z
        return z

    @torch.inference_mode()
    @rank_zero_only
    def sample(
        self,
        prompt,
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        generator=None,
        size=(1024, 1024),
        steps=20,
        guidance_scale=6.5,
    ):
        self.model.eval()
        self.first_stage_model.to(self.target_device)

        model_dtype = next(self.model.parameters()).dtype
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        prompts_batch = {
            "target_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "original_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "crop_coords_top_left": torch.stack([torch.asarray((0, 0))]).cuda(),
        }
        prompts_batch["prompts"] = prompt
        cond = self.encode_batch(prompts_batch)
        prompts_batch["prompts"] = negative_prompt
        uncond = self.encode_batch(prompts_batch)

        crossattn = torch.cat([uncond["crossattn"], cond["crossattn"]], dim=0)
        vector = torch.cat([uncond["vector"], cond["vector"]], dim=0)
        cond = {
            "crossattn": crossattn.cuda().to(model_dtype),
            "vector": vector.cuda().to(model_dtype),
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
            pred_uncond, pred_text = noise_pred.chunk(
                num_latent_input
            )  # uncond by negative prompt
            noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents.cuda()).prev_sample

        latents = self.decode_first_stage(latents.to(torch.float32))
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        self.model.train()
        return image

    def save_checkpoint(self, model_path):
        cfg = self.config.trainer
        string_cfg = OmegaConf.to_yaml(self.config)
        if cfg.get("save_format") == "safetensors":

            model_path += ".safetensors"
            state_dict = self.model.state_dict()
            # check if any keys startswith modules. if so, remove the modules. prefix
            if any([key.startswith("module.") for key in state_dict.keys()]):
                state_dict = {
                    key.replace("module.", ""): value
                    for key, value in state_dict.items()
                }
            save_file(state_dict, model_path, metadata={"trainer_config": string_cfg})
        else:
            model_path += ".ckpt"
            torch.save(
                model_path,
                {"state_dict": self.model.state_dict(), "trainer_config": string_cfg},
            )
        rank_zero_print(f"Saved model to {model_path}")

    def forward(self, batch):
        raise NotImplementedError
