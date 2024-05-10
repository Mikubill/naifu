import safetensors
import torch
import os, math
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger

import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from modules.sdxl_model import StableDiffusionModel
from modules.scheduler_utils import apply_snr_weight
from lightning.pytorch.utilities.model_summary import ModelSummary

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()
    
    params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append(
            {"params": model.conditioner.embedders[0].parameters(), "lr": lr}
        )
        
    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append(
            {"params": model.conditioner.embedders[1].parameters(), "lr": lr}
        )

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(
        params_to_optim, **optim_param
    )
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
    
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_state_dict(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)
        
    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")
        
    model.model, optimizer = fabric.setup(model.model, optimizer)
    if config.advanced.get("train_text_encoder_1") or config.advanced.get("train_text_encoder_2"):
        model.conditioner = fabric.setup(model.conditioner)
        
    dataloader = fabric.setup_dataloaders(dataloader)
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
    if hasattr(fabric.strategy, "_fsdp_kwargs"):
        model._fsdp_engine = fabric.strategy
        
    # set here; 
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = stratified_uniform(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

# min_value = 0.002
# max_value = 700
# image_d = 64
# noise_d_low = 32
# noise_d_high = 64
# sigma_data = 0.5

class EDMPrecond:
    def __init__(self):
        self.sigma_data = 1.0
        self.prediction_type = "v_prediction"
        
    def __call__(self, net, x, sigma, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

        if self.prediction_type == "epsilon":
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        elif self.prediction_type == "v_prediction":
            c_out = -sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        else:
            raise ValueError(f"Prediction type {self.prediction_type} is not supported.")
        
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = net((c_in * x), c_noise.flatten(), **model_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

class EDMLoss:
    def __init__(self):
        # sigma = rand_log_normal(shape=[y.shape[0],], loc=self.P_mean, scale=self.P_std)
        # sigma = rand_cosine_interpolated(
        #     shape=[y.shape[0],], 
        #     image_d=128, 
        #     noise_d_low=32,
        #     noise_d_high=128,
        #     sigma_data=1.0, 
        #     min_value=0.002, 
        #     max_value=120.0
        # )
        self.sample_sigma = lambda y: rand_v_diffusion(
            shape=[y.shape[0],], 
            sigma_data=1.0, 
            min_value=1e-3, #0.002, 
            max_value=1e3, #120.0
        )
        self.sigma_data = 1.0
        self.precond = EDMPrecond()

    def __call__(self, model, y, **model_kwargs):
        sigma = self.sample_sigma(y) 
        sigma = sigma[:, None, None, None].to(y.device)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma
        D_yn = self.precond(model, y + n, sigma, **model_kwargs)
        loss = weight * ((D_yn - y) ** 2)
        return torch.mean(loss, dim=1).mean()

class SupervisedFineTune(StableDiffusionModel):  
    def init_model(self):
        super().init_model()
        
        self.loss_fn = EDMLoss()
        
    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["pixels"].to(self.first_stage_model.dtype))
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        else:
            self.first_stage_model.cpu()
            latents = self._normliaze(batch["pixels"])

        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}
        loss = self.loss_fn(self.model, y=latents, c=cond)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss