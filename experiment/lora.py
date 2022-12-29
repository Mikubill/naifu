import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from lib.model import StableDiffusionModel, get_class
from pytorch_lightning.utilities import rank_zero_only

# (wip) LoRA: https://github.com/cloneofsimo/lora
target_replace_module = ["CrossAttention", "Attention", "GEGLU"]

class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4):
        super().__init__()
        assert r <= min(in_features, out_features), f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1/r)
        nn.init.zeros_(self.lora_up.weight)
        
    def load_linear_weight(self, weight, bias=None):
        self.linear.weight = weight
        if bias is not None:
            self.linear.bias = bias
        
    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale
    
def inject_trainable_lora(model, r=4):
    require_grad_params = []
    names = []

    for module in model.modules():
        if module.__class__.__name__ not in target_replace_module:
            continue 
        for name, child in module.named_modules():
            if child.__class__.__name__ == "Linear":
                loraLinear = LoraInjectedLinear(child.in_features, child.out_features, child.bias is not None, r)
                loraLinear.load_linear_weight(child.weight, child.bias)
                
                require_grad_params.append(loraLinear.lora_up.parameters())
                require_grad_params.append(loraLinear.lora_down.parameters())
                loraLinear.lora_up.weight.requires_grad = True
                loraLinear.lora_down.weight.requires_grad = True
                
                module._modules[name] = loraLinear 
                names.append(name)
                
    return require_grad_params, names

def extract_lora_ups_down(model):
    loras = []
    for module in model.modules():
        if module.__class__.__name__ not in target_replace_module:
            continue 
        for _child_module in module.modules():
            if _child_module.__class__.__name__ == "LoraInjectedLinear":
                loras.append((_child_module.lora_up, _child_module.lora_down))
    if len(loras) == 0:
        raise ValueError("No lora injected.")
    return loras

def load_lora_weight(model, loras=None):
    if loras == None:       
        return
    for module in model.modules():
        if module.__class__.__name__ not in target_replace_module:
            continue 
        for _, child in module.named_modules():
            if child.__class__.__name__ == "LoraInjectedLinear":
                child.lora_up.weight = loras.pop(0)
                child.lora_down.weight = loras.pop(0)

def save_lora_weight(model):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight)
        weights.append(_down.weight)

    return weights

class LoRADiffusionModel(StableDiffusionModel):
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        
    def init_model(self):
        super().init_model()
        self.unet.requires_grad_(False)
        self.params_to_optim, _ = inject_trainable_lora(self.unet, self.config.lora.rank)
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"]["lora"] = save_lora_weight(self.unet)
        
    def on_load_checkpoint(self, checkpoint):
        load_lora_weight(self.unet, checkpoint["state_dict"]["lora"])
        
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
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(self.weight_dtype)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(self.weight_dtype)).sample
        
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
        if self.config.lightning.auto_lr_find:
            self.config.optimizer.params.lr = self.lr
            
        new_lr, scaled = self.get_scaled_lr(self.config.optimizer.params.lr)
        if scaled:
            self.config.optimizer.params.lr = new_lr
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
        
        optimizer = get_class(self.config.optimizer.name)(
            itertools.chain(*self.params_to_optim), 
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
    