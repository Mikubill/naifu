import contextlib
import functools
import gc
import tarfile
from pathlib import Path
from typing import Iterable

import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device('cuda')

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, unet, vae, encoder, noise_scheduler, ema, config):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.config = config
        self.text_encoder = encoder
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32

    def training_step(self, batch, batch_idx):
        # Convert images to latent space
        latent_dist = self.vae.encode(batch["pixel_values"].to(device, dtype=self.weight_dtype)).latent_dist
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
        encoder_hidden_states = self.text_encoder(batch['input_ids'].to(device), output_hidden_states=True)
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
        return optimizer


class EMAModel:
    """
    Maintains (exponential) moving average of a set of parameters.
    Ref: https://github.com/harubaru/waifu-diffusion/diffusers_trainer.py#L478

    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from model.parameters()`).
        decay: The exponential decay.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]

    @contextlib.contextmanager
    def average_parameters(self, parameters):
        r"""
        Context manager for validation/inference with averaged parameters.
        """
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)


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
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    noise_scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler")
    ema = EMAModel(unet.parameters())
    
    weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32
    vae = vae.to(device, dtype=weight_dtype)
    unet = unet.to(device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    ema.to(device, dtype=weight_dtype)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if config.trainer.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    return tokenizer, StableDiffusionModel(unet, vae, text_encoder, noise_scheduler, ema, config)
