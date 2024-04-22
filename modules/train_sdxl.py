import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
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
    
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler

def get_sigmas(sch, timesteps, n_dim=4, dtype=torch.float32, device="cuda:0"):
    sigmas = sch.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = sch.timesteps.to(device)
    timesteps = timesteps.to(device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class SupervisedFineTune(StableDiffusionModel):    
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

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timestep_start = advanced.get("timestep_start", 0)
        timestep_end = advanced.get("timestep_end", 1000)
        do_edm_style_training = advanced.get("do_edm_style_training", False)

        # Sample a random timestep for each image
        if not do_edm_style_training:
            timesteps = torch.randint(
                low=timestep_start, 
                high=timestep_end,
                size=(bsz,),
                dtype=torch.int64,
                device=latents.device,
            )
            timesteps = timesteps.long()
        else:
            # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
            # instead of discrete timesteps, so here we sample indices to get the noise levels
            # from `scheduler.timesteps`
            indices = torch.randint(
                low=timestep_start, 
                high=timestep_end,
                size=(bsz,),
                dtype=torch.int64,
            )
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
                    
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
        # We then precondition the final model inputs based on these sigmas instead of the timesteps.
        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        if do_edm_style_training:
            sigmas = get_sigmas(self.noise_scheduler, timesteps, len(noisy_latents.shape), noisy_latents.dtype, latents.device)
            if "EDM" in self.noise_scheduler.__class__.__name__:
                inp_noisy_latents = self.noise_scheduler.precondition_inputs(noisy_latents, sigmas)
            else:
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
        else:
            inp_noisy_latents = noisy_latents

        # Predict the noise residual
        noise_pred = self.model(inp_noisy_latents, timesteps, cond)
        
        weighting = None
        if do_edm_style_training:
            # Similar to the input preconditioning, the model predictions are also preconditioned
            # on noised model inputs (before preconditioning) and the sigmas.
            # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
            if "EDM" in self.noise_scheduler.__class__.__name__:
                noise_pred = self.noise_scheduler.precondition_outputs(noisy_latents, noise_pred, sigmas)
            else:
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    noise_pred = noise_pred * (-sigmas) + noisy_latents
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    noise_pred = noise_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                        noisy_latents / (sigmas**2 + 1)
                    )
            # We are not doing weighting here because it tends result in numerical problems.
            # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
            # There might be other alternatives for weighting as well:
            # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
            if "EDM" not in self.noise_scheduler.__class__.__name__:
                weighting = (sigmas**-2.0).float()

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        if do_edm_style_training:
            target = latents
        elif is_v:
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        
        min_snr_gamma = advanced.get("min_snr", False)            
        if min_snr_gamma:
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            if min_snr_gamma:
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                
            loss = loss.mean()  # mean over batch dimension
        else:
            if weighting is not None:
                loss = torch.mean(
                    (weighting.float() * (noise_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                )
                loss = loss.mean()
            else:
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
