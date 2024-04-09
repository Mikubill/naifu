import torch
import os
import json
import lightning as pl

from omegaconf import OmegaConf
from common.utils import get_class, EmptyInitWrapper
from common.logging import logger
from modules.sdxl_model import StableDiffusionModel
from lightning.pytorch.utilities.model_summary import ModelSummary

from models.sgm import GeneralConditioner
from modules.scheduler_utils import apply_snr_weight
from modules.sdxl_utils import disabled_train, UnetWrapper, AutoencoderKLWrapper
from modules.config_sdxl_refiner import model_config

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )
    # todo: add to dataset load
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    aesdata = config.dataset.aes_path
    aesdata = json.load(open(config.dataset.aes_path)) if not isinstance(aesdata, int) else aesdata

    def get_batch_extras(self, path):
        if isinstance(aesdata, int):
            return {"aes": aesdata}
        return {"aes": aesdata[path]}
    
    dataset.store.get_batch_extras = get_batch_extras.__get__(dataset.store, dataset_class)
    dataloader = dataset.init_dataloader()
    
    params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder"):
        lr = config.advanced.get("text_encoder_lr", config.optimizer.params.lr)
        params_to_optim.append(
            {"params": model.conditioner.embedders[0].parameters(), "lr": lr}
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
    if config.advanced.get("train_text_encoder"):
        model.conditioner = fabric.setup(model.conditioner)
        
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler

class SupervisedFineTune(StableDiffusionModel):
    def build_models(
        self, 
        init_unet=True, 
        init_vae=True, 
        init_conditioner=True
    ):
        trainer_cfg = self.config.trainer
        config = self.config
        advanced = config.get("advanced", {})

        model_params = model_config.model.params
        if trainer_cfg.use_xformers:
            unet_config = model_params.network_config.params
            vae_config = model_params.first_stage_config.params
            unet_config.spatial_transformer_attn_type = "softmax-xformers"
            vae_config.ddconfig.attn_type = "vanilla-xformers"

        for conditioner in model_params.conditioner_config.params.emb_models:
            if "CLIPEmbedder" not in conditioner.target:
                continue
            self.max_token_length = self.config.dataset.get("max_token_length", 75) + 2
            conditioner.params["device"] = str(self.target_device)
            conditioner.params["max_length"] = self.max_token_length

        tte = advanced.get("train_text_encoder", False)
        model_params.conditioner_config.params.emb_models[0]["is_trainable"] = tte

        self.scale_factor = model_params.scale_factor
        with EmptyInitWrapper(self.target_device):
            vae_config = model_params.first_stage_config.params
            unet_config = model_params.network_config.params
            cond_config = model_params.conditioner_config.params

            vae = AutoencoderKLWrapper(**vae_config) if init_vae else None
            unet = UnetWrapper(unet_config) if init_unet else None
            conditioner = GeneralConditioner(**cond_config) if init_conditioner else None

        vae.eval()
        vae.train = disabled_train
        vae.requires_grad_(False)
        return vae, unet, conditioner
    
    def encode_batch(self, batch):
        # build aes batch
        aesdata = list(map(lambda x: torch.tensor(x["aes"]), batch["extras"]))
        batch["aesthetic_score"] = torch.asarray(aesdata).to(self.target_device)
        self.conditioner.to(self.target_device)
        return self.conditioner(batch)
    
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

        # https://arxiv.org/abs/2301.11706
        if advanced.get("input_perturbation"):
            noise = noise + float(advanced.get("input_perturbation_val")) * torch.randn_like(noise)

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
        noise_pred = self.model(noisy_latents, timesteps, cond)

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
