import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel
from lib.model import StableDiffusionModel, get_class
from torch_ema import ExponentialMovingAverage
from .utils import AbstractTokenizer
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# pip install diffusers accelerate transformers sentencepiece gradio ftfy open-clip-torch
# clip_text: openai/clip-vit-large-patch14
# t5_text: google/flan-t5-base
    
class CustomEncoderDiffusionModel(StableDiffusionModel):
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        
    def init_model(self):      
        config = self.config 
        scheduler_cls = get_class(config.scheduler.name)
        self.noise_scheduler = scheduler_cls(**config.scheduler.params)
        
        self.tokenizer = AbstractTokenizer()
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet") 
             
        self.unet.to(torch.float32)
        if config.trainer.half_encoder:
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
                
        if config.get("sampling"):
            self.pipeline = StableDiffusionPipeline(
                vae=self.vae, 
                text_encoder=self.text_encoder, 
                tokenizer=self._tokenizer,  
                unet=self.unet, 
                scheduler=self.noise_scheduler, 
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            self.pipeline.set_progress_bar_config(disable=True)
        
        # finally setup ema
        if config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
        
    def encode_tokens(self, prompt):
        text_inputs = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        attention_mask = None
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
            output_hidden_states=True
        )        
        return text_embeddings.last_hidden_state