import safetensors
import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from modules.cascade_model import StableCascadeModel, EFFNET_PREPROCESS
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

    if dataset.store.__class__.__name__ == "DirectoryImageStore":
        dataset.store.transforms = EFFNET_PREPROCESS
    else:
        dataset.store.scale_factor = 1.0

    params_to_optim = [{"params": model.stage_c.parameters()}]
    if config.advanced.get("train_text_encoder"):
        lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
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

    model.effnet.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.stage_c, optimizer = fabric.setup(model.stage_c, optimizer)
    if config.advanced.get("train_text_encoder"):
        model.text_encoder = fabric.setup(model.text_encoder)
        
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(StableCascadeModel):
    def get_module(self):
        return self.stage_c
    
    def forward(self, batch):
        if not batch["is_latent"]:
            self.effnet.to(self.target_device)
            latents = self.encode_pixels(batch["pixels"])
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(
                    torch.isnan(latents), torch.zeros_like(latents), latents
                )
        else:
            self.effnet.cpu()
            latents = batch["pixels"]

        model_dtype = next(self.stage_c.parameters()).dtype
        batch_size = latents.size(0)

        hidden, pooled = self.encode_prompts(batch["prompts"])
        hidden = hidden.to(model_dtype)
        pooled = pooled.unsqueeze(1).to(model_dtype)

        image_embed = torch.zeros(1, 768, device=self.target_device)
        image_embed = image_embed.repeat(batch_size, 1, 1)
        with torch.no_grad():
            noised, noise, target, logSNR, noise_cond, loss_weight = self.gdf.diffuse(
                x0=latents, shift=1, loss_shift=1
            )

        pred = self.stage_c.forward(
            x=noised,
            r=noise_cond,
            clip_text=hidden,
            clip_text_pooled=pooled,
            clip_img=image_embed,
        )
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        loss_adjusted = (loss * loss_weight).mean()

        if self.config.adaptive_loss_weight:
            self.gdf.loss_weight.update_buckets(logSNR, loss)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss_adjusted
