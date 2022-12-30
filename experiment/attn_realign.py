
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from lib.model import get_class
from experiment.custom_encoder import CustomEncoderDiffusionModel
from pytorch_lightning.utilities import rank_zero_only

def unfreeze_and_extract_params(*models):
    params_to_optim = []
    first_stage_target =  ["CrossAttention"]
    second_stage_target = ["CrossAttention", "Attention", "GEGLU", "CLIPAttention"]
    third_stage_target = [] # full gradient train
    for module in itertools.chain.from_iterable([model.modules() for model in models]):
        if module.__class__.__name__ in first_stage_target:
            for param in module.parameters():
                param.require_grad = True
            params_to_optim.append(module.parameters())
    print(f"Trainable layers: {len(params_to_optim)}")
    return params_to_optim

    
class AttnRealignModel(CustomEncoderDiffusionModel):     
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
               
    def configure_optimizers(self):
        if self.config.lightning.auto_lr_find:
            self.config.optimizer.params.lr = self.lr
            
        new_lr, scaled = self.get_scaled_lr(self.config.optimizer.params.lr)
        if scaled:
            self.config.optimizer.params.lr = new_lr
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
            
        params_to_optimize = (
            itertools.chain.from_iterable(unfreeze_and_extract_params(self.unet)) 
        )
        optimizer = get_class(self.config.optimizer.name)(
            params_to_optimize, **self.config.optimizer.params
        )
        scheduler = get_class(self.config.lr_scheduler.name)(
            optimizer=optimizer,
            **self.config.lr_scheduler.params
        )
        
        warmup_config = self.config.lr_scheduler.warmup
        if warmup_config.enabled and self.trainer.global_step < warmup_config.num_warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = min(pg["lr"], warmup_config.init_lr)
            
        return [[optimizer], [scheduler]]
        
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

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss