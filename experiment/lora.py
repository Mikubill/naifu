import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from lib.model import StableDiffusionModel, get_class

class LoRABaseModel(torch.nn.Module):
    
    def __init__(self, unet, text_encoder, config):

        super().__init__()
        names = []
        self.multiplier, self.r, alpha = config.multipier, config.rank, config.lora_alpha
        
        self.text_encoder_loras = self.create_modules('lora_te', text_encoder, ["CLIPAttention", "CLIPMLP"], alpha)
        self.unet_loras = self.create_modules('lora_unet', unet, ["Transformer2DModel", "Attention"], alpha)
        
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")
        
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
            
    def create_modules(self, prefix, model, target_replace_module, alpha):
        blocks = []
        for name, module in model.named_modules():
            if module.__class__.__name__ not in target_replace_module:
                continue 
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
                    
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    lora_module = LoRAModule(lora_name, child_module, self.multiplier, self.r, alpha)
                    blocks.append(lora_module)
                    
        return blocks
    
    def inject(self, unet=True, text_encoder=True):
        if not unet:
            self.unet_loras = []
        if not text_encoder:
            self.text_encoder_loras = []
            
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.inject()
            self.add_module(lora.lora_name, lora)
            

class LoRAModule(torch.nn.Module):
    
    def __init__(self, lora_name, base_layer, multiplier=1.0, lora_dim=4, alpha=1):
        super().__init__()
        self.lora_name = lora_name    
        self.lora_dim = lora_dim
        
        if base_layer.__class__.__name__ == 'Conv2d':
            in_dim = base_layer.in_channels
            out_dim = base_layer.out_channels
            self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
            self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        else:
            in_dim = base_layer.in_features
            out_dim = base_layer.out_features
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)
        
        self.multiplier = multiplier
        self.base_layer = base_layer
        
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha))   
        
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
    
    def inject(self):
        # inject and init weights
        self.org_forward = self.base_layer.forward
        self.base_layer.forward = self.forward
        del self.base_layer

    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale


class LoRADiffusionModel(StableDiffusionModel):
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        
    def init_model(self):
        super().init_model()
        
        self.unet.train()
        if self.config.lora.train_text_encoder:
            self.text_encoder.train()
            
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        if self.config.trainer.gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            
        self.lora =  LoRABaseModel(self.unet, self.text_encoder, self.config.lora)
        self.lora.inject(self.config.lora.train_unet, self.config.lora.train_text_encoder)
        self.lora.requires_grad_(True)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.config.lora.lowvram:
            self.unet.to(self.device, dtype=torch.float16)
            self.lora.to(self.device, dtype=torch.float32)
            self.text_encoder.to(self.device, dtype=torch.float16)
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("lora.")}
    
    def training_step(self, batch, batch_idx):
        input_ids, latents = batch[0], batch[1]
        encoder_hidden_states = self.encode_tokens(input_ids).to(self.unet.dtype)
        if not self.dataset.use_latent_cache:
            latents = self.encode_pixels(latents).to(self.unet.dtype)

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
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")  
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        
        enumerate_params = lambda loras: itertools.chain.from_iterable([lora.parameters() for lora in loras])
        params_to_optim = []
        if self.config.lora.train_unet:
            new_unet_lr, scaled = self.get_scaled_lr(self.config.lora.unet_lr)
            if scaled:
                print(f"Using scaled unet LR (LoRA): {new_unet_lr}")
            params_to_optim.append({
                'params': enumerate_params(self.lora.unet_loras),
                'lr': new_unet_lr
            })
                
        if self.config.lora.train_text_encoder:
            new_encoder_lr, scaled = self.get_scaled_lr(self.config.lora.encoder_lr)
            if scaled:
                print(f"Using scaled text_encoder LR (LoRA): {new_encoder_lr}")
            params_to_optim.append({
                'params': enumerate_params(self.lora.text_encoder_loras),
                'lr': new_encoder_lr
            })
        
        optimizer = get_class(self.config.optimizer.name)(
            params_to_optim, 
            **self.config.optimizer.params
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
    
