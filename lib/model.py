
import functools
import math
import os
import tarfile
from pathlib import Path

from data.buckets import AspectRatioSampler
from data.store import ImageStore
from data.store import AspectRatioDataset
from omegaconf import OmegaConf

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
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
    convert_ldm_bert_checkpoint,
    get_world_size
)

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, batch_size):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
        self.lr = self.config.optimizer.params.lr
        self.batch_size = batch_size 
        
    def prepare_data(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        dataset_cls = AspectRatioDataset if self.config.arb.enabled else ImageStore
        
        # init Dataset
        self.dataset = dataset_cls(
            size=self.config.trainer.resolution,
            seed=self.config.trainer.seed,
            rank=local_rank,
            init=not self.config.arb.enabled,
            augconf=self.config.dataset.augment,
            **self.config.dataset
        )
        
        # init sampler
        self.data_sampler = AspectRatioSampler(
            bsz=self.batch_size,
            config=self.config, 
            rank=local_rank, 
            dataset=self.dataset, 
            world_size=get_world_size()
        ) if self.config.arb.enabled else None
        
    def setup(self, stage):
        config = self.config
        scheduler_cls = get_class(config.scheduler.name)
        self.noise_scheduler = scheduler_cls(**config.scheduler.params)
        
        if (Path(self.model_path) / "model.ckpt").is_file():
            # use autoconvert
            self.unet, self.vae, self.text_encoder, self.tokenizer = load_sd_checkpoint(self.model_path)                
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(config.encoder.text if config.encoder.text else self.model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(config.encoder.text if config.encoder.text else self.model_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(config.encoder.vae if config.encoder.vae else self.model_path, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet") 
             
        self.unet.to(self.weight_dtype)
        if config.trainer.half_encoder or self.weight_dtype == torch.float16:
            self.vae.to(torch.float16)
            self.text_encoder.to(torch.float16)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        if config.trainer.gradient_checkpointing: 
            self.unet.enable_gradient_checkpointing()
            
        if config.trainer.use_xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)
        
        # finally setup ema
        if config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
            
        self.dataset.set_tokenizer(self.tokenizer)
        
    def train_dataloader(self):
        self.data_sampler.update_bsz(self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            sampler=self.data_sampler,
            num_workers=self.config.dataset.num_workers,
            persistent_workers=True,
        )
        return dataloader
            
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        input_ids = batch[0]
        z = []
        if input_ids.shape[1] > 77:  
            # todo: Handle end-of-sentence truncation
            while max(map(len, input_ids)) != 0:
                rem_tokens = [x[75:] for x in input_ids]
                tokens = []
                for j in range(len(input_ids)):
                    tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [self.tokenizer.eos_token_id] * 75)

                rebuild = torch.asarray([[self.tokenizer.bos_token_id] + list(x[:75]) + [self.tokenizer.eos_token_id] for x in tokens])
                z.append(rebuild)
                input_ids = rem_tokens
        else:
            z.append(input_ids)
            
        # Get the text embedding for conditioning
        encoder_hidden_states = None
        for tokens in z:
            state = self.text_encoder(tokens.to(self.device), output_hidden_states=True)
            state = self.text_encoder.text_model.final_layer_norm(state['hidden_states'][-self.config.trainer.clip_skip])
            encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)
        
        return encoder_hidden_states, batch[1]
    
    def training_step(self, batch, batch_idx):
        encoder_hidden_states, pixels = batch
        
        # Convert images to latent space
        latent_dist = self.vae.encode(pixels.to(dtype=torch.float16 if self.config.trainer.half_encoder else self.weight_dtype)).latent_dist
        latents = latent_dist.sample() * 0.18215

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
            
        # Scale LR OPs
        f = self.trainer.accumulate_grad_batches * self.config.trainer.init_batch_size * self.trainer.num_nodes * self.trainer.num_devices
        if self.config.trainer.lr_scale == "linear":
            self.config.optimizer.params.lr *= f
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
        elif self.config.trainer.lr_scale == "sqrt":
            self.config.optimizer.params.lr *= math.sqrt(f)
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
        elif self.config.trainer.lr_scale == "none":
            pass
        else:
            raise ValueError(self.config.lr_scale)
        
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
        or not ((Path(model_path) / "model_index.json").is_file() or (Path(model_path) / "model.ckpt").is_file())
    ):
        try:
            Path(model_path).mkdir(exist_ok=True)
            download_model(model_url, model_path)
        except FileNotFoundError:
            pass

    return StableDiffusionModel(model_path, config, config.trainer.init_batch_size)


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