import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import rank_zero_print, get_class
from common.dataset import AspectRatioDataset, worker_init_fn
from modules.cascade_model import StableCascadeModel, EFFNET_PREPROCESS
from lightning.pytorch.utilities.model_summary import ModelSummary


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, config=config, device=fabric.device
    )
    dataset = AspectRatioDataset(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        base_len=config.trainer.resolution,
        **config.dataset,
    )
    if dataset.store.__class__.__name__ == "DirectoryImageStore":
        dataset.store.transforms = EFFNET_PREPROCESS
    else:
        dataset.store.scale_factor = 1.0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=None,
        persistent_workers=False,
        num_workers=config.dataset.get("num_workers", 4),
        worker_init_fn=worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )

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

    model.effnet.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(StableCascadeModel):
    def forward(self, batch):
        if not batch["is_latent"]:
            self.effnet.to(self.target_device)
            latents = self.encode_pixels(batch["pixels"])
            if torch.any(torch.isnan(latents)):
                rank_zero_print("NaN found in latents, replacing with zeros")
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
                x0=latents,
                shift=1,
                loss_shift=1
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
