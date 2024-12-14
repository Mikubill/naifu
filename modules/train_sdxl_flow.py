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


class SupervisedFineTune(StableDiffusionModel):  
    def init_model(self):
        super().init_model()
        
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

        bsz = latents.shape[0]
        noise = torch.randn_like(latents, device=latents.device)
        sigmas = torch.sigmoid(torch.randn((bsz,), device=self.target_device))
        
        timesteps = (sigmas * 1000.0) # maybe better to use discrete timesteps
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        model_pred = self.model(noisy_latents.to(torch.bfloat16), timesteps, cond)

        target = noise - latents
        loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(latents.shape[0], -1), 1) 
        return loss.mean()
 