import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import load_torch_file, get_class
from common.logging import logger
from modules.sdxl_model import StableDiffusionModel
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file
from lightning.pytorch.utilities.model_summary import ModelSummary

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

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
        
    return model, dataset, dataloader, optimizer, scheduler

class SupervisedFineTune(StableDiffusionModel):  
    def init_model(self):
        sd = load_torch_file(self.model_path, self.target_device)
        self.first_stage_model, self.model, self.conditioner = self.build_models()
        
        # original sgxl denoiser
        dn_cfg = self.config.denoiser
        if "edm_vpred.sigma_max" in sd.keys():
            sigma_max = sd["edm_vpred.sigma_max"].item()
            sigma_min = sd["edm_vpred.sigma_min"].item()
            dn_cfg.params.sigma_max = sigma_max
            dn_cfg.params.sigma_min = sigma_min
            self.extras = {
                "edm_vpred.sigma_max": torch.tensor(sigma_max), 
                "edm_vpred.sigma_min": torch.tensor(sigma_min)
            }
            if self.config.sampling.enabled:
                self.config.sampling.scheduler_params.sigma_max = sigma_max
                self.config.sampling.scheduler_params.sigma_min = sigma_min
        
        weighting_params = dn_cfg.get("weighting_params", {})
        sigma_sampler_params = dn_cfg.get("sigma_sampler_params", dn_cfg.params)
        
        self.loss_weighting = get_class(dn_cfg.weighting)(**weighting_params)
        self.denoiser = get_class(dn_cfg.target)(**dn_cfg.params)
        self.sigma_sampler = get_class(dn_cfg.sigma_sampler)(**sigma_sampler_params)
        
        self.to(self.target_device)
        logger.info(f"Loading model from {self.model_path}")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            logger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")
            
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size
            
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
        # timesteps = torch.randint(
        #     advanced.get("timestep_start", 0),
        #     advanced.get("timestep_end", 1000),
        #     (bsz,),
        #     dtype=torch.int64,
        # )
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = self.sigma_sampler(bsz).to(self.target_device)
        sigmas_bc = append_dims(sigmas, latents.ndim)
        noisy_latents = latents + noise * sigmas_bc

        # Predict the noise residual
        model_pred = self.denoiser(self.model, noisy_latents, sigmas, cond)
        w = append_dims(self.loss_weighting(sigmas), latents.ndim)
        
        # Calculate the loss
        loss = torch.mean((w * (model_pred - latents) ** 2).reshape(noise.shape[0], -1), 1)
        loss = loss.mean()
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
    
    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        cfg = self.config.trainer
        state_dict = self.state_dict()
        state_dict.update(self.extras)
        # check if any keys startswith modules. if so, remove the modules. prefix
        if any([key.startswith("module.") for key in state_dict.keys()]):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                
        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
        else:
            state_dict = {"state_dict": state_dict, **metadata} 
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")
