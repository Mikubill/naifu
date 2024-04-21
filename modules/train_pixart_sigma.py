import torch
import os
import lightning as pl

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKL

import torch.distributed as dist
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only

from common.utils import load_torch_file, get_class
from common.logging import logger

from models.pixart.sigma import DiT_XL_2, get_model_kwargs, sample
from models.pixart.diffusion import (
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from models.pixart.diffusion import ModelVarType, ModelMeanType, LossType


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = DiffusionModel(model_path=model_path, config=config, device=fabric.device)
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()

    params_to_optim = [{"params": model.model.parameters()}]
    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )

    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.model, optimizer = fabric.setup(model.model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
    if hasattr(fabric.strategy, "_fsdp_kwargs"):
        model._fsdp_engine = fabric.strategy
        
    # set here; 
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler


class DiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def init_model(self):
        t5_model_path = self.config.trainer.get(
            "t5_model_path", "PixArt-alpha/PixArt-XL-2-1024-MS"
        )
        vae_model_path = self.config.trainer.get(
            "vae_model_path", "stabilityai/sdxl-vae"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            t5_model_path, legacy=False, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            t5_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            subfolder="text_encoder",
        )
        self.vae = AutoencoderKL.from_pretrained(vae_model_path)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        dit_state_dict = load_torch_file(self.model_path)
        betas = get_named_beta_schedule(
            schedule_name="linear", num_diffusion_timesteps=1000
        )
        timesteps = space_timesteps(num_timesteps=1000, section_counts=[1000])
        self.diffusion = SpacedDiffusion(
            use_timesteps=timesteps,
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.LEARNED_RANGE,
            loss_type=LossType.MSE,
            snr=False,
            return_startx=False,
            # rescale_timesteps=rescale_timesteps,
        )

        base = self.config.trainer.resolution
        self.model = DiT_XL_2(input_size=base // 8, interpolation_scale=base // 512, max_token_length=300)
        self.model.to(memory_format=torch.channels_last).train()

        logger.info("Loading weights from checkpoint: DiT-XL-2-1024-MS.pth")
        result = self.model.load_state_dict(dit_state_dict, strict=False)
        print(result)

    @torch.no_grad()
    def encode_tokens(self, prompts):
        self.text_encoder.to(self.target_device)
        with torch.autocast("cuda", enabled=False):
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=300,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_attention_mask = text_inputs.attention_mask
            prompt_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids.to(self.target_device), 
                attention_mask=prompt_attention_mask.to(self.target_device),
                return_dict=True
            )['last_hidden_state']
            return prompt_embeds, prompt_attention_mask

    def forward(self, batch):
        prompts = batch["prompts"]
        prompt_embeds, prompt_attention_mask = self.encode_tokens(prompts)

        if not batch["is_latent"]:
            self.vae.to(self.target_device)
            latent_dist = self.vae.encode(batch["pixels"]).latent_dist
            latents = latent_dist.sample() * self.vae.config.scaling_factor
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        else:
            self.vae.cpu()
            latents = batch["pixels"] * self.vae.config.scaling_factor

        model_dtype = next(self.model.parameters()).dtype
        bsz = latents.shape[0]

        # Forward pass through the model
        timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
        latents = latents.to(model_dtype)
        model_kwg = get_model_kwargs(latents, self.model)
        loss = self.diffusion.training_losses(
            self.model,
            latents,
            timesteps,
            model_kwargs=dict(
                y=prompt_embeds.to(self.target_device, dtype=model_dtype),
                mask=prompt_attention_mask.to(self.target_device, dtype=model_dtype),
                **model_kwg,
            ),
        )["loss"].mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss

    def generate_samples(self, logger, current_epoch, global_step):
        if self._fabric_wrapped.world_size > 2:
            return self.generate_samples_dist(logger, current_epoch, global_step)
        return self.generate_samples_normal(logger, current_epoch, global_step)

    def generate_samples_dist(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        rank = 0
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
        current_device = self.vae.device
        self.vae.to(self.target_device)
        
        local_prompts = prompts[rank::world_size]
        for idx, prompt in tqdm(
            enumerate(local_prompts), desc=f"Sampling (Process {rank})", total=len(local_prompts), leave=False
        ):
            image = sample(
                model=self.model,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=[prompt],
                negative_prompt="bad quality, low quality",
                size=size,
                generator=generator,
                device=self.target_device,
            )
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_p{rank}_{idx}.png"
            )
            images.append((image[0], prompt))

        gathered_images = [images]
        if dist.is_initialized() and world_size > 1:
            gathered_images = [None] * world_size
            dist.all_gather_object(gathered_images, images)
            
        if rank in [0, -1]:
            all_images = []
            all_prompts = []
            for entry in gathered_images:
                if isinstance(entry, list):
                    entry = entry[0]
                imgs, prompts = entry
                all_prompts.append(prompts)
                all_images.append(imgs)

            
            if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
                logger.log_image(
                    key="samples", images=all_images, caption=all_prompts, step=global_step
                )
                
        self.vae.to(current_device)
        self.model.train()
                
    @rank_zero_only
    def generate_samples_normal(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()
        current_device = self.vae.device
        self.vae.to(self.target_device)
        
        for idx, prompt in tqdm(enumerate(prompts), desc="Sampling", leave=False, total=len(prompts)):
            image = sample(
                model=self.model,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=[prompt],
                negative_prompt="bad quality, low quality",
                size=size,
                generator=generator,
                device=self.target_device,
            )
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_{idx}.png"
            )
            images.extend(image)

        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(key="samples", images=images, caption=prompts, step=global_step)
        
        self.vae.to(current_device)
        self.model.train()

    def save_checkpoint(self, model_path, metadata):
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                weight_to_save = self.model._forward_module.state_dict()
        else:
            weight_to_save = self.model.state_dict()
                
        self._save_checkpoint(model_path, weight_to_save, metadata)

    @rank_zero_only
    def _save_checkpoint(self, model_path, weight_to_save, metadata):
        cfg = self.config.trainer
        state_dict = weight_to_save
        # check if any keys startswith modules. if so, remove the modules. prefix
        if any([key.startswith("module.") for key in state_dict.keys()]):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            
        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
        else:
            model_path += ".ckpt"
            state_dict = {"state_dict": state_dict, **metadata},
            torch.save(state_dict, model_path)
            
        logger.info(f"Saved model to {model_path}")
