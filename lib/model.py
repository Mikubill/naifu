
import functools
import math
import os
import tarfile
import tempfile

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from pathlib import Path
from tqdm.auto import tqdm
from data.store import AspectRatioDataset, ImageStore
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from lib.utils import get_local_rank, get_world_size, min_snr_weighted_loss
from lib.utils import convert_to_sd, convert_to_df

# define the LightningModule


class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, batch_size):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.lr = self.config.optimizer.params.lr
        self.batch_size = batch_size
        self.init_model()

    def init_model(self):
        config = self.config
        scheduler_cls = DDIMScheduler

        if Path(self.model_path).is_file():
            # use autoconvert
            from_safetensors = Path(self.model_path).with_suffix(".safetensors").is_file()
            if from_safetensors:
                from safetensors import safe_open

                checkpoint = {}
                with safe_open(self.model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint[key] = f.get_tensor(key)
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint = torch.load(self.model_path, map_location=device)

            # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
            # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
            while "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            self.pipeline = convert_to_df(checkpoint, return_pipe=True)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(self.model_path)

        self.pipeline.set_progress_bar_config(disable=True)
        self.unet, self.vae, self.text_encoder, self.tokenizer, self.noise_scheduler = \
            self.pipeline.unet, self.pipeline.vae, self.pipeline.text_encoder, self.pipeline.tokenizer, self.pipeline.scheduler
        self.unet.to(self.device, dtype=torch.float32)
        self.unet.train()

        try:
            torch.compile(self.unet, mode="max-autotune", fullgraph=True, dynamic=True)
        except Exception as e:
            print(f"Skip: unable to compile model - {e}")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if config.trainer.get("train_text_encoder"):
            self.text_encoder.train()
            self.text_encoder.requires_grad_(True)

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

        self.use_latent_cache = self.config.dataset.get("cache_latents")

    def setup(self, stage):
        local_rank = get_local_rank()
        world_size = get_world_size()
        dataset_cls = AspectRatioDataset if self.config.arb.enabled else ImageStore

        arb_config = {
            "bsz": self.config.trainer.batch_size,
            "seed": self.config.trainer.seed,
            "world_size": world_size,
            "global_rank": local_rank,
            **self.config.arb
        }

        # init Dataset
        self.dataset = dataset_cls(
            arb_config=arb_config,
            size=self.config.trainer.resolution,
            seed=self.config.trainer.seed,
            rank=local_rank,
            init=not self.config.arb.enabled,
            tokenizer=self.tokenizer,
            **self.config.dataset
        )

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            num_workers=1,
            batch_size=self.batch_size,
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
        if not self.use_latent_cache:
            latents = self.encode_pixels(latents).to(self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.trainer.get("offset_noise"):
            noise = torch.randn_like(latents) + float(self.config.trainer.get("offset_noise_val")) \
                * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)

        # https://arxiv.org/abs/2301.11706
        if self.config.trainer.get("input_perturbation"):
            noise = noise + float(self.config.trainer.get("input_perturbation_val")) * torch.randn_like(noise)

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

        if not self.config.trainer.get("min_snr"):
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        else:
            gamma = self.config.trainer.get("min_snr_val")
            loss = min_snr_weighted_loss(noise_pred.float(), target.float(), timesteps, self.noise_scheduler, gamma=gamma)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        # Logging to TensorBoard by default
        major, minor, _ = pl.__version__.split('.')
        if int(major) >= 2:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("train_loss", loss)

        return loss

    def get_scaled_lr(self, base):
        # Scale LR OPs
        f = self.trainer.accumulate_grad_batches * self.config.trainer.batch_size * self.trainer.num_nodes * self.trainer.num_devices
        if self.config.trainer.lr_scale == "linear":
            return base * f, True
        elif self.config.trainer.lr_scale == "sqrt":
            return base * math.sqrt(f), True
        elif self.config.trainer.lr_scale == "none":
            return base, False
        else:
            raise ValueError(self.config.lr_scale)

    def configure_optimizers(self):
        # align LR with batch size in case we're using lrfinder
        if self.lr != self.config.optimizer.params.lr:
            self.config.optimizer.params.lr = self.lr

        new_lr, scaled = self.get_scaled_lr(self.config.optimizer.params.lr)
        if scaled:
            self.config.optimizer.params.lr = new_lr
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))

        params_to_optim = [{'params': self.unet.parameters()}]
        if self.config.trainer.get("train_text_encoder") == True:
            text_encoder_group = {'params': self.text_encoder.parameters()}
            if self.config.trainer.get("text_encoder_lr"):
                text_encoder_group['lr'], te_scaled = self.get_scaled_lr(self.config.trainer.get("text_encoder_lr"))
            params_to_optim.append(text_encoder_group)

        optimizer = get_class(self.config.optimizer.name)(
            params_to_optim, **self.config.optimizer.params
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

        if self.use_latent_cache:
            self.dataset.cache_latents(self.vae, self.data_sampler.buckets if self.config.arb.enabled else None, self.config)

    def on_train_epoch_start(self) -> None:
        if self.use_latent_cache:
            self.vae.to("cpu")

    def on_train_batch_end(self, *args, **kwargs):
        if self.config.trainer.use_ema:
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        state_dict = convert_to_sd(checkpoint["state_dict"])
        if self.config.checkpoint.get("extended"):
            if self.config.checkpoint.extended.save_fp16_weights:
                state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}

        checkpoint["state_dict"] = state_dict

        if self.config.trainer.use_ema:
            checkpoint["model_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        unet_sd, vae_sd, te_sd = convert_to_df(checkpoint["state_dict"])
        self.unet.load_state_dict(unet_sd)
        self.vae.load_state_dict(vae_sd)
        self.text_encoder.load_state_dict(te_sd)
        checkpoint["state_dict"] = {}
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
