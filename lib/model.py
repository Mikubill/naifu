import functools
import tarfile
from pathlib import Path

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from .ema import ExponentialMovingAverage
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config):
        super().__init__()
        self.config = config
            
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        self.noise_scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler")
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        if self.config.trainer.gradient_checkpointing: 
            self.unet.enable_gradient_checkpointing()
        
    def training_step(self, batch, batch_idx):
        
        # latent conversion not requires graident
        with torch.no_grad():
            # Convert images to latent space
            latent_dist = self.vae.encode(batch[1]).latent_dist
            latents = latent_dist.sample() * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch[0], output_hidden_states=True)
        encoder_hidden_states = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-self.config.trainer.clip_skip])

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")  
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
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
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
        
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
        or not (Path(model_path) / "model_index.json").is_file()
    ):
        Path(model_path).mkdir(exist_ok=True)
        download_model(model_url, model_path)

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    return tokenizer, StableDiffusionModel(model_path, config)
