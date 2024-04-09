import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class
from common.logging import logger
from modules.scheduler_utils import apply_snr_weight
from modules.sd_model_diffusers import StableDiffusionModel
from lightning.pytorch.utilities.model_summary import ModelSummary


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, config=config, device=fabric.device
    )
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()

    params_to_optim = [{"params": model.unet.parameters()}]
    # params_to_optim = [{'params': model.model.parameters()}]
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

    model.vae.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.unet, optimizer = fabric.setup(model.unet, optimizer)
    if config.advanced.get("train_text_encoder"):
        model.text_encoder = fabric.setup(model.text_encoder)
        
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(StableDiffusionModel):
    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.vae.to(self.target_device)
            latents = self.encode_pixels(batch["pixels"])
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(
                    torch.isnan(latents), torch.zeros_like(latents), latents
                )
        else:
            self.vae.cpu()
            latents = batch["pixels"] * self.vae.config.scaling_factor

        self.text_encoder.to(self.target_device)
        model_dtype = next(self.unet.parameters()).dtype
        hidden_states = self.encode_prompts(batch["prompts"])

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=latents.device
            )
            noise_scale = float(advanced.get("offset_noise_val"))
            noise = torch.randn_like(latents) + noise_scale * offset
            

        # https://arxiv.org/abs/2301.11706
        if advanced.get("input_perturbation"):
            interp = advanced.get("input_perturbation_val")
            noise = noise + float(interp) * torch.randn_like(noise)

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            advanced.get("timestep_start", 0),
            advanced.get("timestep_end", 1000),
            (bsz,),
            dtype=torch.int64,
            device=latents.device,
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=hidden_states,
        ).sample

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        target = noise if not is_v else self.noise_scheduler.get_velocity(latents, noise, timesteps)

        min_snr_gamma = advanced.get("min_snr", False)
        if min_snr_gamma:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            if min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                
            loss = loss.mean()  # mean over batch dimension
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
