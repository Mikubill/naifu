import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel
from lib.model import StableDiffusionModel, get_class

# pip install diffusers accelerate transformers sentencepiece gradio ftfy open-clip-torch
# clip_text: openai/clip-vit-large-patch14
# t5_text: google/flan-t5-base
    
class MixinModel(StableDiffusionModel):        
    def setup(self, stage):
        super().setup(stage)
        self.unet_prior = [UNet2DConditionModel.from_pretrained("/root/workspace/animesfw/unet")]
        assert self.noise_scheduler.config.prediction_type == "epsilon"
        
    def training_step(self, batch, batch_idx):
        input_ids, pixels = batch[0], batch[1]
        encoder_hidden_states = self.encode_tokens(input_ids)
        latents = self.encode_pixels(pixels)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        noise_pred_prior = self.unet_prior[0](noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + F.mse_loss(noise_pred.float(), noise_pred_prior.float(), reduction="mean") * 0.5
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss