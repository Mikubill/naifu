import torch
import json
import torch.utils.checkpoint
import lightning as pl

from pathlib import Path
from common.logging import logger
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler, DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only
from tqdm import tqdm
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
        p = StableDiffusionPipeline
        if Path(self.model_path).is_file():
            self.pipeline = pipeline = p.from_single_file(self.model_path)
        else:
            self.pipeline = pipeline = p.from_pretrained(self.model_path)
            
        self.vae, self.unet = pipeline.vae, pipeline.unet
        self.text_encoder, self.tokenizer = (
            pipeline.text_encoder,
            pipeline.tokenizer,
        )
        self.max_prompt_length = 225 + 2
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.scale_factor = 0.13025

        self.unet.enable_gradient_checkpointing()
        if trainer_cfg.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if advanced.get("train_text_encoder"):
            self.text_encoder.requires_grad_(True)
            self.text_encoder.gradient_checkpointing_enable()

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        self.max_token_length = self.config.dataset.get("max_token_length", 75) + 2
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size
        
        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)
        cache_snr_values(self.noise_scheduler, self.target_device)

    def get_module(self):
        return self.unet
    
    def encode_pixels(self, inputs):
        feed_pixel_values = inputs
        latents = []
        for i in range(0, feed_pixel_values.shape[0], self.vae_encode_bsz):
            with torch.autocast("cuda", enabled=False):
                lat = self.vae.encode(feed_pixel_values[i : i + self.vae_encode_bsz]).latent_dist.sample()
            latents.append(lat)
        latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        return latents

    def encode_prompts(self, prompts):   
        self.text_encoder.to(self.target_device)         
        input_ids = self.tokenizer(
            prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_token_length,
            return_tensors="pt"
        ).input_ids 
        tokenizer_max_length = self.tokenizer.model_max_length
        oids = []
        for iids in input_ids:
            z = []
            for i in range(1, self.max_token_length - tokenizer_max_length + 2, tokenizer_max_length - 2):  # (1, 152, 75)
                ids_chunk = (
                    iids[0].unsqueeze(0),
                    iids[i : i + tokenizer_max_length - 2],
                    iids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                z.append(ids_chunk)
            oids.append(torch.stack(z))
            
        oids = torch.stack(oids)
        bs = oids.size(0)
        input_ids = oids.reshape((-1, tokenizer_max_length))
        
        state = self.text_encoder(input_ids.to(self.target_device), output_hidden_states=True)
        encoder_hidden_states = state['hidden_states'][-self.config.trainer.clip_skip]
        if self.config.trainer.clip_skip > 1:
            encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
        
        encoder_hidden_states = encoder_hidden_states.reshape((bs, -1, encoder_hidden_states.shape[-1]))
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, self.max_token_length, tokenizer_max_length):
            states_list.append(encoder_hidden_states[:, i : i + tokenizer_max_length - 2])  
            
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        encoder_hidden_states = torch.cat(states_list, dim=1)
        return encoder_hidden_states

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
        size=(512, 512),
        steps=20,
        guidance_scale=6.5,
    ):
        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8) 
        size = (height, width)
        self.vae.to(self.target_device)
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        pipeline = StableDiffusionPipeline(
            unet=self.unet,
            vae=self.vae,
            text_encoder_1=self.text_encoder_1,
            text_encoder_2=self.text_encoder_2,
            noise_scheduler=scheduler,
        )
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
