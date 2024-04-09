import torch
import torch.utils.checkpoint
import lightning as pl

from pathlib import Path
from common.logging import logger
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler, DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only

from tqdm import tqdm
from modules.sdxl_utils import get_hidden_states_sdxl
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def init_model(self):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})
        
        logger.info(f"Loading model from {self.model_path}")
        p = StableDiffusionXLPipeline
        if Path(self.model_path).is_file():
            self.pipeline = pipeline = p.from_single_file(self.model_path)
        else:
            self.pipeline = pipeline = p.from_pretrained(self.model_path)
          
        self.vae, self.unet = pipeline.vae, pipeline.unet
        self.text_encoder_1, self.text_encoder_2, self.tokenizer_1, self.tokenizer_2 = (
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.tokenizer,
            pipeline.tokenizer_2,
        )
        self.max_prompt_length = 225 + 2
        self.vae.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.scale_factor = 0.13025

        self.unet.enable_gradient_checkpointing()
        if trainer_cfg.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if advanced.get("train_text_encoder_1"):
            self.text_encoder_1.requires_grad_(True)
            self.text_encoder_1.gradient_checkpointing_enable()

        if advanced.get("train_text_encoder_2"):
            self.text_encoder_2.requires_grad_(True)
            self.text_encoder_2.gradient_checkpointing_enable()

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size
        
        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)
            
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
            self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)
        
        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)
        
    def get_module(self):
        return self.unet
    
    def _denormlize(self, latents):
        scaling_factor = self.vae.config.scaling_factor
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = latents * self.latents_std / scaling_factor + self.latents_mean
        else:
            latents = 1.0 / scaling_factor * latents
        return latents
    
    def _normliaze(self, latents):
        scaling_factor = self.vae.config.scaling_factor
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = (latents - self.latents_mean) * scaling_factor / self.latents_std
        else:
            latents = scaling_factor * latents
        return latents
             
    def encode_pixels(self, inputs):
        feed_pixel_values = inputs
        latents = []
        for i in range(0, feed_pixel_values.shape[0], self.vae_encode_bsz):
            with torch.autocast("cuda", enabled=False):
                lat = self.vae.encode(feed_pixel_values[i : i + self.vae_encode_bsz]).latent_dist.sample()
            latents.append(lat)
        latents = torch.cat(latents, dim=0)
        return self._normliaze(latents)
        
    def encode_prompt(self, batch):
        prompt = batch["prompts"]
        hidden_states1, hidden_states2, pool2 = get_hidden_states_sdxl(
            prompt,
            self.max_prompt_length,
            self.tokenizer_1,
            self.tokenizer_2,
            self.text_encoder_1,
            self.text_encoder_2,
        )
        text_embedding = torch.cat([hidden_states1, hidden_states2], dim=2)
        return text_embedding, pool2
    
    def compute_time_ids(self, original_size, crops_coords_top_left, target_size):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = torch.cat([original_size, crops_coords_top_left, target_size])
        add_time_ids = add_time_ids.to(self.target_device)
        return add_time_ids
    
    @rank_zero_only
    def generate_samples(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))

        for idx, prompt in tqdm(enumerate(prompts), desc="Sampling", leave=False):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(Path(config.save_dir) / f"sample_e{current_epoch}_s{global_step}_{idx}.png")
            images.extend(image)

        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(key="samples", images=images, caption=prompts, step=global_step)

    @torch.inference_mode()
    def sample(
        self,
        prompt,
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        generator=None,
        size=(1024, 1024),
        steps=20,
        guidance_scale=6.5,
    ):
        self.vae.to(self.target_device)
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        pipeline = StableDiffusionXLPipeline(
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer_1,
            tokenizer_2=self.tokenizer_2,
            scheduler=scheduler,
        )
        pipeline.set_progress_bar_config(disable=True)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=size[0],
            height=size[1],
            steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            return_dict=False,
        )[0]
        return image

    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        self.pipeline.save_pretrained(model_path)
        logger.info(f"Saved model to {model_path}")

    def forward(self, batch):
        raise NotImplementedError
