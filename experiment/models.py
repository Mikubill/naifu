import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel
from lib.model import StableDiffusionModel, get_class
from torch_ema import ExponentialMovingAverage
from .encoder import FrozenOpenCLIPEmbedder
from .utils import FrozenCustomEncoder

# pip install diffusers accelerate transformers sentencepiece gradio ftfy open-clip-torch
# clip_text: openai/clip-vit-large-patch14
# t5_text: google/flan-t5-base
    
class MultiEncoderDiffusionModel(StableDiffusionModel):
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        self.config = config
        self.model_path = model_path
        self.weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
        self.lr = self.config.optimizer.params.lr
        self.batch_size = batch_size 
        
    def setup(self, stage):      
        config = self.config 
        scheduler_cls = get_class(config.scheduler.name)
        self.noise_scheduler = scheduler_cls(**config.scheduler.params)
        
        self.tokenizer = FrozenCustomEncoder(FrozenOpenCLIPEmbedder(device="cpu"))
        self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet") 
             
        self.unet.to(self.weight_dtype)
        if config.trainer.half_encoder or self.weight_dtype == torch.float16:
            self.vae.to(torch.float16)

        self.vae.requires_grad_(False)
        if config.trainer.gradient_checkpointing: 
            self.unet.enable_gradient_checkpointing()
            
        if config.trainer.get("use_xformers") == True:
            if hasattr(self.unet, "set_use_memory_efficient_attention_xformers"):
                self.unet.set_use_memory_efficient_attention_xformers(True)
            elif hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                self.unet.enable_xformers_memory_efficient_attention()
        
        if config.trainer.get("attention_slicing") == True:
            if hasattr(self.unet, "enable_attention_slicing"):
                self.unet.enable_attention_slicing()
        
        # finally setup ema
        if config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
            
        self.dataset.set_tokenizer(self.tokenizer)
            
    def encode_tokens(self, prompt):
        return prompt