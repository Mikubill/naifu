import bisect
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Union

import diffusers
import torch
import torch.nn as nn
import torch.utils.checkpoint

class LoConBaseModel(torch.nn.Module):
    
    def __init__(self, unet, config):

        super().__init__()
        names = []
        self.multiplier, self.r, alpha, dropout = config.multipier, config.rank, config.lora_alpha, config.get("dropout", 0.0)
        self.conv_r, self.conv_alpha = getattr(config, "conv_rank", self.r), getattr(config, "conv_alpha", alpha)
        
        # te_modules = [
        #     "CLIPAttention", 
        #     "CLIPMLP"
        # ]
        # self.text_encoder_loras = self.create_modules('lora_te', text_encoder, te_modules, alpha, dropout)
        self.text_encoder_loras = []
        
        unet_modules = [
            "SpatialTransformer", 
            "ResBlock", 
            "Downsample", 
            "Upsample"
        ]
        self.unet_loras = self.create_modules('lora_unet', unet, unet_modules, alpha, dropout)
            
        # print(f"create LoCon for Text Encoder: {len(self.text_encoder_loras)} modules.")
        print(f"create LoCon for U-Net: {len(self.unet_loras)} modules.")
        
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
            
    def create_modules(self, prefix, model, target_replace_module, alpha, dropout):
        blocks = []
        for name, module in model.named_modules():
            if module.__class__.__name__ in target_replace_module:
                for child_name, child_module in module.named_modules():
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    
                    if child_module.__class__.__name__ == "Linear":
                        lora_module = LoConModule(lora_name, child_module, self.multiplier, self.r, alpha, dropout)
                    elif child_module.__class__.__name__ == "Conv2d":
                        k_size, *_ = child_module.kernel_size
                        if k_size == 1:
                            lora_module = LoConModule(lora_name, child_module, self.multiplier, self.r, alpha, dropout)
                        else:
                            lora_module = LoConModule(lora_name, child_module, self.multiplier, self.conv_r, self.conv_alpha, dropout)
                    else:
                        continue
                    
                    blocks.append(lora_module)
            elif name in target_replace_module:
                lora_name = prefix + '.' + name 
                lora_name = lora_name.replace('.', '_')
                if module.__class__.__name__ == "Linear":
                    lora_module = LoConModule(lora_name, module, self.multiplier, self.r, alpha, dropout)
                elif module.__class__.__name__ == "Conv2d":
                    k_size, *_ = module.kernel_size
                    if k_size == 1:
                        lora_module = LoConModule(lora_name, module, self.multiplier, self.r, alpha, dropout)
                    else:
                        lora_module = LoConModule(lora_name, module, self.multiplier, self.conv_r, self.conv_alpha, dropout)
                else:
                    continue
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
            

class LoConModule(torch.nn.Module):
    def __init__(self, lora_name, base_layer, multiplier=1.0, lora_dim=4, alpha=1, dropout=0):
        super().__init__()
        self.lora_name = lora_name    
        self.lora_dim = lora_dim
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        
        if base_layer.__class__.__name__ == 'Conv2d':
            in_dim = base_layer.in_channels
            k_size = base_layer.kernel_size
            stride = base_layer.stride
            padding = base_layer.padding
            out_dim = base_layer.out_channels
            self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
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

    @torch.enable_grad() 
    def forward(self, x):
        return self.org_forward(x) + self.dropout(self.lora_up(self.lora_down(x))) * self.multiplier * self.scale