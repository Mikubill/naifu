from pathlib import Path
import torch
import torch.utils.checkpoint
import lightning as pl
from PIL import Image

import torch
from tqdm import tqdm
import torch.distributed as dist

from models.sgm import GeneralConditioner
from modules.sdxl_utils import disabled_train, UnetWrapper, AutoencoderKLWrapper
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
from common.utils import get_class, load_torch_file, EmptyInitWrapper, get_world_size
from common.logging import logger

from diffusers import DDPMScheduler
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file
from modules.config_sdxl_base import model_config

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def build_models(self, init_unet=True, init_vae=True, init_conditioner=True):
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

        if advanced.get("use_checkpoint", True) == False:
            model_params.network_config.params.use_checkpoint = False

        tte_1 = advanced.get("train_text_encoder_1", False)
        tte_2 = advanced.get("train_text_encoder_2", False)
        model_params.conditioner_config.params.emb_models[0]["is_trainable"] = tte_1
        model_params.conditioner_config.params.emb_models[1]["is_trainable"] = tte_2

        with EmptyInitWrapper(self.target_device):
            vae_config = model_params.first_stage_config.params
            unet_config = model_params.network_config.params
            cond_config = model_params.conditioner_config.params
            unet = UnetWrapper(unet_config) if init_unet else None

        vae = AutoencoderKLWrapper(**vae_config) if init_vae else None
        conditioner = GeneralConditioner(**cond_config) if init_conditioner else None    
        self.scale_factor = advanced.get("scale_factor", model_params.scale_factor)            
        if advanced.get("latents_mean", None):
            self.latents_mean = torch.tensor(advanced.latents_mean)
            self.latents_std = torch.tensor(advanced.latents_std)
            self.latents_mean = self.latents_mean.view(1, 4, 1, 1).to(self.target_device)
            self.latents_std = self.latents_std.view(1, 4, 1, 1).to(self.target_device)

        vae.eval()
        vae.train = disabled_train
        vae.requires_grad_(False)
        return vae, unet, conditioner

    def init_model(self):
        advanced = self.config.get("advanced", {})
        sd = load_torch_file(self.model_path, self.target_device)
        self.first_stage_model, self.model, self.conditioner = self.build_models()
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

        # allow custom class
        if self.config.get("noise_scheduler"):
            scheduler_cls = get_class(self.config.noise_scheduler.name)
            self.noise_scheduler = scheduler_cls(**self.config.noise_scheduler.params)

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

        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)

        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)

    def get_module(self):
        return self.model

    def encode_batch(self, batch):
        self.conditioner.to(self.target_device)
        return self.conditioner(batch)
    
    def _denormlize(self, latents):
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = latents * self.latents_std / self.scale_factor + self.latents_mean
        else:
            latents = 1.0 / self.scale_factor * latents
        return latents
    
    def _normliaze(self, latents):
        if hasattr(self, "latents_mean"):
            # https://github.com/huggingface/diffusers/pull/7111
            latents = (latents - self.latents_mean) * self.scale_factor / self.latents_std
        else:
            latents = self.scale_factor * latents
        return latents

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = self._denormlize(z)
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        latents = []
        self.first_stage_model = self.first_stage_model.float()
        with torch.autocast("cuda", enabled=False):
            for i in range(0, x.shape[0], self.vae_encode_bsz):
                o = x[i : i + self.vae_encode_bsz]
                latents.append(self.first_stage_model.encode(o).sample())
        z = torch.cat(latents, dim=0)
        return self._normliaze(z)

    def generate_samples(self, logger, current_epoch, global_step):
        if hasattr(self, "_fabric_wrapped"):
            if self._fabric_wrapped.world_size > 2:
                self.generate_samples_dist(logger, current_epoch, global_step)
                return self._fabric_wrapped.barrier()
                
        return self.generate_samples_seq(logger, current_epoch, global_step)

    def generate_samples_dist(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        rank = 0
        world_size = self._fabric_wrapped.world_size
        rank = self._fabric_wrapped.global_rank

        local_prompts = prompts[rank::world_size]
        for idx, prompt in tqdm(
            enumerate(local_prompts), desc=f"Sampling (Process {rank})", total=len(local_prompts), leave=False
        ):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_p{rank}_{idx}.png"
            )
            images.append((image[0], prompt))

        gathered_images = [None] * world_size
        dist.all_gather_object(gathered_images, images)
        
        self.model.train()
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
    
    @rank_zero_only
    def generate_samples_seq(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        for idx, prompt in tqdm(
            enumerate(prompts), desc="Sampling", total=len(prompts), leave=False
        ):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_{idx}.png"
            )
            images.extend(image)

        self.model.train()
        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(
                key="samples", images=images, caption=prompts, step=global_step
            )

    @torch.inference_mode()
    def sample(
        self,
        prompt,
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        generator=None,
        size=(1024, 1024),
        steps=25,
        guidance_scale=6.5,
    ):
        self.first_stage_model.to(self.target_device)

        model_dtype = next(self.model.parameters()).dtype
        scheduler_name = self.config.sampling.get(
            "scheduler", "diffusers.EulerDiscreteScheduler"
        )
        scheduler_params = self.config.sampling.get(
            "scheduler_params",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
                num_train_timesteps=1000,
            ),
        )
        if self.config.advanced.get("v_parameterization", False):
            scheduler_params["prediction_type"] = "v_prediction"

        scheduler_cls = get_class(scheduler_name)
        scheduler = scheduler_cls(**scheduler_params)
        prompts_batch = {
            "target_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "original_size_as_tuple": torch.stack([torch.asarray(size)]).cuda(),
            "crop_coords_top_left": torch.stack([torch.asarray((0, 0))]).cuda(),
        }
        prompts_batch["prompts"] = prompt
        cond = self.encode_batch(prompts_batch)
        prompts_batch["prompts"] = negative_prompt
        uncond = self.encode_batch(prompts_batch)

        crossattn = torch.cat([uncond["crossattn"], cond["crossattn"]], dim=0)
        vector = torch.cat([uncond["vector"], cond["vector"]], dim=0)
        cond = {
            "crossattn": crossattn.cuda().to(model_dtype),
            "vector": vector.cuda().to(model_dtype),
        }

        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)
        size = (height, width)
        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        num_latent_input = 2
        
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.cuda().to(model_dtype)

            noise_pred = self.model(latent_model_input, torch.asarray([t]).cuda(), cond)
            pred_uncond, pred_text = noise_pred.chunk(
                num_latent_input
            )  # uncond by negative prompt
            noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents.cuda()).prev_sample

        latents = self.decode_first_stage(latents.to(torch.float32))
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image

    def save_checkpoint(self, model_path, metadata):
        weight_to_save = None
        if hasattr(self, "_fsdp_engine"):
            from lightning.fabric.strategies.fsdp import _get_full_state_dict_context
            
            weight_to_save = {}    
            world_size = self._fsdp_engine.world_size
            with _get_full_state_dict_context(self.model._forward_module, world_size=world_size):
                unet_weight = self.model._forward_module.state_dict()
                for key in unet_weight.keys():
                    weight_to_save[f"model.{key}"] = unet_weight[key]
                
            with _get_full_state_dict_context(self.conditioner._forward_module, world_size=world_size):
                cond_weight = self.conditioner._forward_module.state_dict()
                for key in cond_weight.keys():
                    weight_to_save[f"conditioner.{key}"] = cond_weight[key]
                
            vae_weight = self.first_stage_model.state_dict()
            for key in vae_weight.keys():
                weight_to_save[f"first_stage_model.{key}"] = vae_weight[key]
        elif hasattr(self, "_deepspeed_engine"):
            from deepspeed import zero
            weight_to_save = {}
            with zero.GatheredParameters(self.model.parameters()):
                unet_weight = self.model.state_dict()
                for key in unet_weight.keys():
                    weight_to_save[f"model.{key}"] = unet_weight[key]
                
            with zero.GatheredParameters(self.conditioner.parameters()):
                cond_weight = self.conditioner.state_dict()
                for key in cond_weight.keys():
                    weight_to_save[f"conditioner.{key}"] = cond_weight[key]
                
            vae_weight = self.first_stage_model.state_dict()
            for key in vae_weight.keys():
                weight_to_save[f"first_stage_model.{key}"] = vae_weight[key]
                
        else:
            weight_to_save = self.state_dict()
                
        self._save_checkpoint(model_path, weight_to_save, metadata)

    @rank_zero_only
    def _save_checkpoint(self, model_path, state_dict, metadata):
        cfg = self.config.trainer
        # check if any keys startswith modules. if so, remove the modules. prefix
        if any([key.startswith("module.") for key in state_dict.keys()]):
            state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }

        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
        else:
            state_dict = {"state_dict": state_dict, **metadata}
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")

    def forward(self, batch):
        raise NotImplementedError
