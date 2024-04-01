import torch
import os
import lightning as pl
from omegaconf import OmegaConf

import copy
import torch.nn.functional as F
from common.utils import get_class
from common.logging import logger
from modules.sdxl_model import StableDiffusionModel
from modules.sdxl_dpo_diffusers import setup_hf_dataloader
from lightning.pytorch.utilities.model_summary import ModelSummary


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, config=config, device=fabric.device
    )
    model._fabric = fabric

    dataset, dataloader = setup_hf_dataloader(config)
    params_to_optim = [{"params": model.model.parameters()}]
    # params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder_1.parameters(), "lr": lr})

    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.text_encoder_2.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )

    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.model, optimizer = fabric.setup(model.model, optimizer)
    if config.advanced.get("train_text_encoder_1") or config.advanced.get("train_text_encoder_2"):
        model.conditioner = fabric.setup(model.conditioner)
        
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(StableDiffusionModel):
    def init_model(self):
        super().init_model()
        # clone frozen unet as ref
        self.unet_ref = copy.deepcopy(self.model)
        self.unet_ref.eval().requires_grad_(False)
        # since we're in dpo, unet_ref in 16bit is ok
        self.unet_ref.to(torch.float16)
        
    def encode_batch(self, batch):
        self.conditioner.to(self.target_device)
        return self.conditioner(batch)
        
    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        self.first_stage_model.to(self.target_device)

        feed_pixel_values = torch.cat(batch["pixels"].chunk(2, dim=1))
        latents = self.encode_first_stage(feed_pixel_values.to(self.first_stage_model.dtype))
        if torch.any(torch.isnan(latents)):
            logger.info("NaN found in latents, replacing with zeros")
            latents = torch.where(
                torch.isnan(latents), torch.zeros_like(latents), latents
            )

        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {
            "crossattn": cond["crossattn"].to(model_dtype).repeat(2, 1, 1),
            "vector": cond["vector"].to(model_dtype).repeat(2, 1),
        }

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype).chunk(2)[0].repeat(2, 1, 1, 1)
        bsz = latents.shape[0] // 2

        # Sample a random timestep for each image
        timesteps = torch.randint(
            advanced.get("timestep_start", 0),
            advanced.get("timestep_end", 1000),
            (bsz,),
            dtype=torch.int64,
            device=latents.device,
        ).repeat(2)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.model(noisy_latents, timesteps, cond)

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        target = noise if not is_v else self.noise_scheduler.get_velocity(latents, noise, timesteps)

        # Compute losses.
        model_losses = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)

        # For logging
        raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
        model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)

        with torch.no_grad():
            ref_d = next(self.unet_ref.parameters()).dtype
            cond_d = {k: v.to(ref_d) for k, v in cond.items()}
            ref_preds = self.unet_ref(noisy_latents.to(ref_d), timesteps, cond_d)
            ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
            ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

            ref_losses_w, ref_losses_l = ref_loss.chunk(2)
            ref_diff = ref_losses_w - ref_losses_l
            raw_ref_loss = ref_loss.mean()
        
        inside_term = -1 * self.config.dpo_beta * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(inside_term).mean()

        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)
        self._fabric.log_dict(
            {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
            }
        )
        return loss
