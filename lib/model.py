
import functools
import math
import os
import tarfile
from pathlib import Path
import tempfile

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from data.buckets import AspectRatioSampler
from data.store import AspectRatioDataset, ImageStore
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm
from transformers import BertTokenizerFast, CLIPTextModel, CLIPTokenizer
from lib.utils import get_local_rank, get_world_size

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, batch_size):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.lr = self.config.optimizer.params.lr
        self.batch_size = batch_size 
        self.save_hyperparameters(config)
        self.init_model()
        
    def init_model(self):
        config = self.config
        scheduler_cls = DDIMScheduler
            
        if (Path(self.model_path) / "model.ckpt").is_file():
            # use autoconvert
            self.unet, self.vae, self.text_encoder, self.tokenizer, self.noise_scheduler = load_sd_checkpoint(self.model_path)   
        else:
            if hasattr(scheduler_cls, "from_pretrained"):
                self.noise_scheduler = scheduler_cls.from_pretrained(self.model_path, subfolder="scheduler")
            else:
                self.noise_scheduler = scheduler_cls.from_config(self.model_path, subfolder="scheduler")
            self.tokenizer = CLIPTokenizer.from_pretrained(config.encoder.text if config.encoder.text else self.model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(config.encoder.text if config.encoder.text else self.model_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(config.encoder.vae if config.encoder.vae else self.model_path, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet") 
         
        self.unet.to(self.device, dtype=torch.float32)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
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
        
        if config.get("sampling"):
            self.pipeline = StableDiffusionPipeline(
                vae=self.vae, 
                text_encoder=self.text_encoder, 
                tokenizer=self.tokenizer, 
                unet=self.unet, 
                scheduler=self.noise_scheduler, 
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False
            )
            self.pipeline.set_progress_bar_config(disable=True)
        
    def prepare_data(self):
        for k, entry in enumerate(self.config.dataset.img_path):
            if entry.startswith("https://") or entry.startswith("http://"):
                dlpath = os.path.join(tempfile.gettempdir(), f"dataset-{k}")
                Path(dlpath).mkdir(exist_ok=True)
                download(entry, dlpath)
                self.config.dataset.img_path[k] = dlpath
            
    def setup(self, stage):
        for k, entry in enumerate(self.config.dataset.img_path):
            if entry.startswith("https://") or entry.startswith("http://"):
                dlpath = os.path.join(tempfile.gettempdir(), f"dataset-{k}")
                self.config.dataset.img_path[k] = dlpath
            
        local_rank = get_local_rank()
        world_size = get_world_size()
        dataset_cls = AspectRatioDataset if self.config.arb.enabled else ImageStore
        
        # init Dataset
        self.dataset = dataset_cls(
            size=self.config.trainer.resolution,
            seed=self.config.trainer.seed,
            rank=local_rank,
            init=not self.config.arb.enabled,
            tokenizer=self.tokenizer,
            **self.config.dataset
        )
        
        # init sampler
        self.data_sampler = None
        if self.config.arb.enabled:
            self.data_sampler = AspectRatioSampler(
                bsz=self.batch_size,
                config=self.config, 
                rank=local_rank, 
                dataset=self.dataset, 
                world_size=world_size,
            ) 
        
    def train_dataloader(self):
        if self.data_sampler:
            self.data_sampler.update_bsz(self.batch_size)
            
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            sampler=self.data_sampler,
            num_workers=self.config.dataset.num_workers,
            batch_size=1 if self.data_sampler else self.batch_size,
            persistent_workers=True,
        )
        return dataloader
    
    def encode_tokens(self, input_ids):
        z = []
        if input_ids.shape[1] > 77:  
            # todo: Handle end-of-sentence truncation
            while max(map(len, input_ids)) != 0:
                rem_tokens = [x[75:] for x in input_ids]
                tokens = []
                for j in range(len(input_ids)):
                    tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [self.tokenizer.eos_token_id] * 75)

                rebuild = [[self.tokenizer.bos_token_id] + list(x[:75]) + [self.tokenizer.eos_token_id] for x in tokens]
                if hasattr(torch, "asarray"):
                    z.append(torch.asarray(rebuild))
                else:
                    z.append(torch.IntTensor(rebuild))
                input_ids = rem_tokens
        else:
            z.append(input_ids)

        # Get the text embedding for conditioning
        encoder_hidden_states = None
        for tokens in z:
            state = self.text_encoder(tokens.to(self.device), output_hidden_states=True)
            state = self.text_encoder.text_model.final_layer_norm(state['hidden_states'][-self.config.trainer.clip_skip])
            encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)
        
        return encoder_hidden_states
    
    def encode_pixels(self, pixels):
        pixels = pixels.to(self.vae.dtype)
        if self.config.trainer.get("vae_slicing"):
            result = []
            for nx in range(pixels.shape[0]):
                px = pixels[nx, ...].unsqueeze(0)
                latent_dist = self.vae.encode(px).latent_dist
                latents = latent_dist.sample() * 0.18215
                result.append(latents)
        
            result = torch.stack(result).squeeze(1)
            return result
        
        # Convert images to latent space
        latent_dist = self.vae.encode(pixels).latent_dist
        latents = latent_dist.sample() * 0.18215
        return latents
            
    def training_step(self, batch, batch_idx):
        input_ids, latents = batch[0], batch[1]
        encoder_hidden_states = self.encode_tokens(input_ids).to(self.unet.dtype)
        if not self.dataset.use_latent_cache:
            latents = self.encode_pixels(latents).to(self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

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
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def get_scaled_lr(self, base):
        # Scale LR OPs
        f = self.trainer.accumulate_grad_batches * self.config.trainer.init_batch_size * self.trainer.num_nodes * self.trainer.num_devices
        if self.config.trainer.lr_scale == "linear":
            return base * f, True
        elif self.config.trainer.lr_scale == "sqrt":
            return base * math.sqrt(f), True
        elif self.config.trainer.lr_scale == "none":
            return base, False
        else:
            raise ValueError(self.config.lr_scale)
    
    def configure_optimizers(self):
        if self.config.lightning.auto_lr_find:
            self.config.optimizer.params.lr = self.lr
            
        new_lr, scaled = self.get_scaled_lr(self.config.optimizer.params.lr)
        if scaled:
            self.config.optimizer.params.lr = new_lr
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
        
        optimizer = get_class(self.config.optimizer.name)(
            self.unet.parameters(), **self.config.optimizer.params
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
    
    def lr_scheduler_step(self, *args):
        warmup_config = self.config.lr_scheduler.warmup
        if not warmup_config.enabled or self.trainer.global_step > warmup_config.num_warmup:
            super().lr_scheduler_step(*args)
                
    def optimizer_step(self, epoch, batch_idx, optimizer, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, *args, **kwargs)
        
        warmup_config = self.config.lr_scheduler.warmup
        if warmup_config.enabled and self.trainer.global_step < warmup_config.num_warmup:
            f = min(1.0, float(self.trainer.global_step + 1) / float(warmup_config.num_warmup))
            if warmup_config.strategy == "cos":
                f = (math.cos(math.pi*(1+f))+1)/2.
            delta = self.config.optimizer.params.lr-warmup_config.init_lr
            for pg in optimizer.param_groups:
                if pg["lr"] >= warmup_config.init_lr:
                    pg["lr"] = warmup_config.init_lr+f*delta
    
    def on_train_start(self):
        if self.config.trainer.use_ema: 
            self.ema.to(self.device, dtype=self.unet.dtype)
            
        if self.config.dataset.get("cache_latents") == True:
            self.dataset.cache_latents(
                self.vae, 
                self.data_sampler.buckets if self.config.arb.enabled else None,
                self.config
            )
            
    def on_train_epoch_start(self) -> None:
        if self.config.dataset.get("cache_latents") == True:
            self.vae.to("cpu")
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.config.trainer.use_ema:
            self.ema.update()
            
    def on_save_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            checkpoint["model_ema"] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            self.ema.load_state_dict(checkpoint["model_ema"])
            

def download(url, model_path="model"):
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
    
    if (
        not Path(model_path).is_dir()
        or not ((Path(model_path) / "model_index.json").is_file() or (Path(model_path) / "model.ckpt").is_file())
    ):
        model_url = config.trainer.model_url
        try:
            Path(model_path).mkdir(exist_ok=True)
            download(model_url, model_path)
        except FileNotFoundError:
            pass

    return StableDiffusionModel(model_path, config, config.trainer.init_batch_size)


def load_sd_checkpoint(model_path):
    from lib.utils import (
        convert_ldm_bert_checkpoint,
        convert_ldm_clip_checkpoint,
        convert_ldm_openclip_checkpoint,
        convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint,
        create_ldm_bert_config, create_unet_diffusers_config,
        create_vae_diffusers_config
    )
    
    vae_path = Path(model_path) / "model.vae.pt"
    checkpoint_path = Path(model_path) / "model.ckpt"
    config_path = Path(model_path) / "config.yaml"
            
    print(f"Loading StableDiffusionModel from {checkpoint_path}")
    original_config = OmegaConf.load(config_path)
    
    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="scaled_linear",
    )
            
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
    elif text_model_type == "FrozenOpenCLIPEmbedder":
        text_encoder = convert_ldm_openclip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    else:
        text_config = create_ldm_bert_config(original_config)
        text_encoder = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
    return unet, vae, text_encoder, tokenizer, scheduler