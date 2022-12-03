
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import AutoencoderKL, UNet2DConditionModel
from lib.model import get_class
from torch_ema import ExponentialMovingAverage
from transformers import  CLIPTextModel, CLIPTokenizer
from transformers import T5EncoderModel, T5Tokenizer

# pip install diffusers accelerate transformers sentencepiece gradio ftfy
# clip_text: openai/clip-vit-large-patch14
# t5_text: google/flan-t5-base

class AbstractTokenizer():
    def __call__(self, x, *args, **kwds):
        return x
    
    def pad(self, x, *args, **kwds):
        return x
    
# define the LightningModule
class T5CLIPDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config):
        super().__init__()
        self.config = config
        self.weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
        
        scheduler_cls = get_class(config.scheduler.name)
        self.noise_scheduler = scheduler_cls(**config.scheduler.params)
        
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.t5_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        self.tokenizer = AbstractTokenizer()
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet") 
             
        self.unet.to(self.weight_dtype)
        if config.trainer.half_encoder or self.weight_dtype == torch.float16:
            self.vae.to(torch.float16)
            self.text_encoder.to(torch.float16)

        self.vae.requires_grad_(False)
        self.clip_encoder.requires_grad_(False)
        self.t5_encoder.requires_grad_(False)
        
        if self.config.trainer.gradient_checkpointing: 
            self.unet.enable_gradient_checkpointing()
            
        if self.config.trainer.use_xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)
        
        # finally setup ema
        if self.config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
            

        self.cond_dim = 768
        self.max_text_len = 77
        self.null_text_hidden = nn.Parameter(torch.randn(1, self.time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned
        self.non_attn_cond_fcn = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.time_cond_dim),
            nn.SiLU(),
            nn.Linear(self.time_cond_dim, self.time_cond_dim)
        )
            
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        prompt, pixels = batch
        
        clip_ids = self.clip_tokenizer(prompt, padding="max_length", truncation=True, max_length=77).input_ids 
        clip_encoded = self.clip_encoder(clip_ids.to(self.clip_encoder.device), output_hidden_states=True)
        clip_state = self.clip_encoder.text_model.final_layer_norm(clip_encoded['hidden_states'][-self.config.trainer.clip_skip])
        
        t5_ids = self.t5_tokenizer(prompt, padding="max_length", truncation=True, max_length=113).input_ids 
        t5_state = self.t5_encoder(t5_ids)
        return torch.cat([clip_state, t5_state, self.null_text_hidden], dim=-2), pixels, clip_state
    
    def training_step(self, batch, batch_idx):
        encoder_hidden_states, pixels, clip_state = batch
        non_attn_cond = self.non_attn_cond_fcn(clip_state.mean(dim=-2))
        
        # Convert images to latent space
        latent_dist = self.vae.encode(pixels.to(dtype=torch.float16 if self.config.trainer.half_encoder else self.weight_dtype)).latent_dist
        latents = latent_dist.sample() * 0.18215
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(self.weight_dtype)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(self.weight_dtype), non_attn_cond).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")  
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.config.lightning.auto_lr_find:
            self.config.optimizer.params.lr = self.lr
            
        optimizer = get_class(self.config.optimizer.name)(
            self.unet.parameters(), **self.config.optimizer.params
        )
        scheduler = get_class(self.config.lr_scheduler.name)(
            optimizer=optimizer,
            **self.config.lr_scheduler.params
        )
        return [[optimizer], [scheduler]]
    
    def on_train_start(self):
        if self.config.trainer.use_ema: 
            self.ema.to(self.device, dtype=self.weight_dtype)
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.config.trainer.use_ema:
            self.ema.update()
            
    def on_save_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            checkpoint["model_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            self.ema.load_state_dict(checkpoint["model_ema"])
