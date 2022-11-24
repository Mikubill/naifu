
import functools
import tarfile
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .ema import ExponentialMovingAverage
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, BertTokenizerFast
from diffusers import AutoencoderKL, UNet2DConditionModel
from lib.utils import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
    create_ldm_bert_config,
    convert_ldm_bert_checkpoint
)

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config):
        super().__init__()
        self.config = config
        self.weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
        scheduler_cls = get_class(config.scheduler.name)
        self.noise_scheduler = scheduler_cls(**config.scheduler.params)
        
        if (Path(model_path) / "model.ckpt").is_file():
            # use autoconvert
            self.unet, self.vae, self.tokenizer, self.text_encoder = load_sd_checkpoint(model_path)                
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet") 
             
        self.unet.to(self.weight_dtype)
        if config.trainer.half_encoder or self.weight_dtype == torch.float16:
            self.vae.to(torch.float16)
            self.text_encoder.to(torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        if self.config.trainer.gradient_checkpointing: 
            self.unet.enable_gradient_checkpointing()
            
        if self.config.trainer.use_xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)
        
        # finally setup ema
        if self.config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
    
    def training_step(self, batch, batch_idx):
        # Convert images to latent space
        latent_dist = self.vae.encode(batch[1].to(dtype=torch.float16 if self.config.trainer.half_encoder else self.weight_dtype)).latent_dist
        latents = latent_dist.sample() * 0.18215
        
        # Get the text embedding for conditioning
        encoder_hidden_states = None
        for tokens in batch[0]:
            state = self.text_encoder(tokens.to(self.device), output_hidden_states=True)
            state = self.text_encoder.text_model.final_layer_norm(state['hidden_states'][-self.config.trainer.clip_skip])
            encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)

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


def download_model(url, model_path="model"):
    print(f'Downloading: "{url}" to {model_path}\n')
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get("content-length", 0))

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        file = tarfile.open(fileobj=r_raw, mode="r|gz")
        file.extractall(path=model_path)


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def load_model(model_path, config):
    model_url = config.trainer.model_url

    if (
        not Path(model_path).is_dir()
        or not (
            (Path(model_path) / "model_index.json").is_file() or (Path(model_path) / "model.ckpt").is_file()
        )
    ):
        Path(model_path).mkdir(exist_ok=True)
        download_model(model_url, model_path)

    model = StableDiffusionModel(model_path, config)
    return model.tokenizer, model


def load_sd_checkpoint(model_path):
    vae_path = Path(model_path) / "model.vae.pt"
    checkpoint_path = Path(model_path) / "model.ckpt"
    config_path = Path(model_path) / "config.yaml"
            
    print(f"Loading StableDiffusionModel from {checkpoint_path}")
    original_config = OmegaConf.load(config_path)
            
    checkpoint = torch.load(checkpoint_path)
    checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            
    vae_checkpoint = checkpoint
    if vae_path.is_file():
        vae_checkpoint = torch.load(vae_path)["state_dict"]

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, path=checkpoint_path, extract_ema=False)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_checkpoint, vae_config)
    
    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
            
    text_model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    if text_model_type == "FrozenCLIPEmbedder":
        text_encoder = convert_ldm_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    else:
        text_config = create_ldm_bert_config(original_config)
        text_encoder = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
    return unet, vae, text_encoder, tokenizer