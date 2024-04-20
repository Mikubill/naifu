import torch
import os
import lightning as pl

from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only

from common.utils import load_torch_file, get_class
from common.logging import logger

from nyanko.naifu.models.pixart.sigma_mp import DiT_XL_2, get_model_kwargs
from nyanko.naifu.models.pixart.diffusion import (
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from nyanko.naifu.models.pixart.diffusion import ModelVarType, ModelMeanType, LossType


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
        self.model = DiT_XL_2(input_size=base // 8, interpolation_scale=base // 512)
        self.model.to(memory_format=torch.channels_last).train()

        logger.info("Loading weights from checkpoint: DiT-XL-2-1024-MS.pth")
        result = self.model.load_state_dict(dit_state_dict, strict=False)
        assert result.unexpected_keys == [
            "pos_embed"
        ], f"Unexpected keys: {result.unexpected_keys}, Missing keys: {result.missing_keys}"

    @torch.no_grad()
    def encode_tokens(self, prompts):
        with torch.autocast("cuda", enabled=False):
            text_encoder = self.text_encoder
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=120,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            prompt_attention_mask = text_inputs.attention_mask.to(text_encoder.device)

            prompt_embeds = text_encoder(
                text_input_ids, attention_mask=prompt_attention_mask
            )[0]
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
                encoder_mask=prompt_attention_mask.to(
                    self.target_device, dtype=model_dtype
                ),
                **model_kwg,
            ),
        )["loss"].mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss

    @rank_zero_only
    def generate_samples(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))

        for idx, prompt in tqdm(enumerate(prompts), desc="Sampling", leave=False):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_{idx}.png"
            )
            images.extend(image)

        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(key="samples", images=images, caption=prompts, step=global_step)

    @torch.no_grad()
    def sample(
        self,
        prompt="1girl, solo",
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        size=(1152, 832),
        steps=28,
        guidance_scale=6.5,
    ):
        self.model.eval()

        prompt = (
            prompt.replace("_", " ")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
            .lower()
        )
        scheduler = DPMSolverMultistepScheduler()
        prompt_embeds, prompt_attention_mask = self.encode_tokens([negative_prompt, prompt])

        latents = torch.randn(1, 4, size[0] // 8, size[1] // 8, device=self.target_device)
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        latents = latents * scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model(
                x=latent_model_input,
                t=torch.stack([t] * latents.shape[0]).to(latents.device),
                y=prompt_embeds.to(latents.device, dtype=latents.dtype),
                encoder_mask=prompt_attention_mask.to(
                    latents.device, dtype=latents.dtype
                ),
                **get_model_kwargs(latents, self.model),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # uncond by negative prompt
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            noise_pred = noise_pred.chunk(2, dim=1)[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        image = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0).detach().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        self.model.train()
        return image
    
    def save_checkpoint(self, model_path, metadata):
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                weight_to_save = self.model._forward_module.state_dict()
        else:
            weight_to_save = self.state_dict()
                
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
