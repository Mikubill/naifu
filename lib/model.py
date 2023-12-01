
import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from pathlib import Path
from data.store import AspectRatioDataset, ImageStore
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch_ema import ExponentialMovingAverage
from lib.utils import get_local_rank, get_world_size
from lib.utils import convert_to_sd, convert_to_df, rank_zero_print
from lightning.pytorch.utilities import rank_zero_only

# define the LightningModule

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

def get_pipeline(model_path):
    if Path(model_path).is_file():
        # use autoconvert
        from_safetensors = Path(model_path).suffix == ".safetensors"
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() == 1 else "cpu"
        if from_safetensors:
            from safetensors import safe_open

            checkpoint = {}
            with safe_open(model_path, framework="pt", device=device) as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(model_path, map_location=device)

        # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
        # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
        while "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        pipeline = convert_to_df(checkpoint, return_pipe=True)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_path)
        
    return pipeline

def min_snr_weighted_loss(eps_pred:torch.Tensor, eps:torch.Tensor, timesteps, noise_scheduler, snr_gamma):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2

    mse_loss_weights = (
        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    )
    # We first calculate the original loss. Then we mean over the non-batch dimensions and
    # rebalance the sample-wise losses with their respective loss weights.
    # Finally, we take the mean of the rebalanced loss.
    loss = F.mse_loss(eps_pred.float(), eps.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    loss = loss.mean()
    return loss


class StableDiffusionModel(pl.LightningModule):
    def __init__(self, pipeline, config, batch_size=0):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.lr = self.config.optimizer.params.lr
        self.batch_size = batch_size if batch_size > 0 else self.config.trainer.batch_size
        self.init_model()
        
    def init_model(self):
        config = self.config
        self.pipeline.set_progress_bar_config(disable=True)
        
        self.is_sdxl = False
        if isinstance(self.pipeline, StableDiffusionPipeline):
            self.unet, self.vae, self.text_encoder = self.pipeline.unet, self.pipeline.vae, self.pipeline.text_encoder
            self.tokenizer, self.noise_scheduler = self.pipeline.tokenizer, self.pipeline.scheduler
        else:
            self.unet, self.noise_scheduler = self.pipeline.unet, self.pipeline.scheduler
            self.vae, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2 = \
                self.pipeline.vae, self.pipeline.text_encoder, self.pipeline.text_encoder_2, self.pipeline.tokenizer, self.pipeline.tokenizer_2
            self.is_sdxl = True

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        self.unet.to(self.device)
        self.unet.train()

        try:
            torch.compile(self.unet, mode="max-autotune", fullgraph=True, dynamic=True)
        except Exception as e:
            rank_zero_print(f"Skip compiling: {e}")

        if hasattr(self, "vae"):
            self.vae.requires_grad_(False)
            
        if config.trainer.get("train_text_encoder"):
            self.text_encoder.train()
            self.text_encoder.requires_grad_(True)
            if self.is_sdxl:
                self.text_encoder_2.train()
                self.text_encoder_2.requires_grad_(True)
        else:
            self.text_encoder.requires_grad_(False)
            if self.is_sdxl:
                self.text_encoder_2.requires_grad_(False)

        if config.trainer.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if config.trainer.get("use_xformers") == True:
            if hasattr(self.unet, "set_use_memory_efficient_attention_xformers"):
                self.unet.set_use_memory_efficient_attention_xformers(True)
            elif hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                self.unet.enable_xformers_memory_efficient_attention()

        if config.trainer.get("attention_slicing") == True:
            if hasattr(self.unet, "enable_attention_slicing"):
                self.unet.enable_attention_slicing()

        # finally setup ema
        if config.trainer.use_ema:
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)

        self.vae_scale_factor = 0.18215 if not self.is_sdxl else 0.13025
        self.use_latent_cache = self.config.dataset.get("cache_latents")

    def setup(self, stage):
        local_rank = get_local_rank()
        world_size = get_world_size()
        arb_config = {
            "bsz": self.config.trainer.batch_size,
            "seed": self.config.trainer.seed,
            "world_size": world_size,
            "global_rank": local_rank,
        }
        if self.config.get("arb", None) is None:
            # calculate arb from resolution
            dataset_cls = AspectRatioDataset
            base = self.config.trainer.resolution
            c_size = 1.5 if base < 1024 else 1
            c_mult = 2
            arb_config.update({
                "base_res": (base, base),
                "max_size": (int(base*c_size), base),
                "divisible": 64,
                "max_ar_error": 4,
                "min_dim": int(base // c_mult),
                "dim_limit": int(base * c_mult),
                "debug": False
            })
        else:
            dataset_cls = AspectRatioDataset if self.config.arb.enabled else ImageStore
            arb_config.update(**self.config.arb)

        # init Dataset
        self.dataset = dataset_cls(
            arb_config=arb_config,
            size=self.config.trainer.resolution,
            seed=self.config.trainer.seed,
            rank=local_rank,
            **self.config.dataset
        )

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            num_workers=1,
            batch_size=self.batch_size,
            persistent_workers=True,
        )
        return dataloader
    
    def encode_tokens_xl(self, batch):
        time_ids_list = []
        for o, t, c in zip(batch["original_size_as_tuple"], batch["target_size_as_tuple"], batch["crop_coords_top_left"]):
            add_time_ids = torch.tensor([list(o + t + c)])
            time_ids_list.append(add_time_ids)
        
        time_ids_list = torch.cat(time_ids_list, dim=0).to(self.device)
        text_encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
        tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2]
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            tokenizer = tokenizers[i]
            text_input_ids = tokenizer(
                batch["prompts"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            text_encoder.to(self.device)
            prompt_embeds = text_encoder(
                text_input_ids.to(self.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1).to(self.unet.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1).to(self.unet.dtype)
        return prompt_embeds, pooled_prompt_embeds, time_ids_list

    def encode_tokens(self, prompts, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
            
        input_ids = self.tokenizer(prompts, padding="do_not_pad", truncation=True, max_length=225).input_ids 
        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        
        z = []
        if input_ids.shape[1] > 77:
            # todo: Handle end-of-sentence truncation
            while max(map(len, input_ids)) != 0:
                rem_tokens = [x[75:] for x in input_ids]
                tokens = []
                for j in range(len(input_ids)):
                    tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [self.tokenizer.eos_token_id] * 75)

                rebuild = [[self.tokenizer.bos_token_id] + list(x[:75]) + [self.tokenizer.eos_token_id] for x in tokens]
                if hasattr(torch, "asarray"):
                    z.append(torch.asarray(rebuild))
                else:
                    z.append(torch.IntTensor(rebuild))
                input_ids = rem_tokens
        else:
            z.append(input_ids)

        # Get the text embedding for conditioning
        encoder_hidden_states = None
        for tokens in z:
            state = self.text_encoder(tokens.to(self.device), output_hidden_states=True)
            state = self.text_encoder.text_model.final_layer_norm(state['hidden_states'][-self.config.trainer.clip_skip])
            encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)

        return encoder_hidden_states

    def encode_pixels(self, pixels):
        vae = self.pipeline.vae
        vae.to(self.device)
        pixels = pixels.to(vae.dtype)
        if self.config.trainer.get("vae_slicing"):
            result = []
            for nx in range(pixels.shape[0]):
                px = pixels[nx, ...].unsqueeze(0)
                latent_dist = vae.encode(px).latent_dist
                latents = latent_dist.sample() * self.vae_scale_factor
                result.append(latents)

            result = torch.stack(result).squeeze(1)
            return result

        # Convert images to latent space
        latent_dist = vae.encode(pixels).latent_dist
        latents = latent_dist.sample() * self.vae_scale_factor
        return latents

    def training_step(self, batch, batch_idx):
        prompts, latents = batch["prompts"], batch["pixel_values"]
        if self.is_sdxl:
            prompt_embeds, pooled_prompt_embeds, time_ids_list = self.encode_tokens_xl(batch)
        else:
            encoder_hidden_states = self.encode_tokens(prompts)
            encoder_hidden_states = encoder_hidden_states.to(self.unet.dtype)
            
        if not self.use_latent_cache:
            latents = self.encode_pixels(latents)
        
        # Cast to the correct dtype
        latents = latents.to(self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.trainer.get("offset_noise"):
            noise = torch.randn_like(latents) + float(self.config.trainer.get("offset_noise_val")) \
                * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)

        # https://arxiv.org/abs/2301.11706
        if self.config.trainer.get("input_perturbation"):
            noise = noise + float(self.config.trainer.get("input_perturbation_val")) * torch.randn_like(noise)

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        if self.is_sdxl:
            unet_added_conditions = {
                "time_ids": time_ids_list,
                "text_embeds": pooled_prompt_embeds
            }
            noise_pred = self.unet(
                noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
            ).sample
        else:
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if not self.config.trainer.get("min_snr"):
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        else:
            gamma = self.config.trainer.get("min_snr_val")
            loss = min_snr_weighted_loss(noise_pred.float(), target.float(), timesteps, self.noise_scheduler, gamma)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        # Logging to TensorBoard by default
        major, minor = pl.__version__.split('.')[:2]
        if int(major) >= 2:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("train_loss", loss)

        return loss

    def get_scaled_lr(self, base):
        # Scale LR OPs
        f = self.trainer.accumulate_grad_batches * self.config.trainer.batch_size * self.trainer.num_nodes * self.trainer.num_devices
        if self.config.trainer.lr_scale == "linear":
            return base * f, True
        elif self.config.trainer.lr_scale == "sqrt":
            return base * math.sqrt(f), True
        elif self.config.trainer.lr_scale == "none":
            return base, False
        else:
            raise ValueError(self.config.lr_scale)

    def configure_optimizers(self):
        # align LR with batch size in case we're using lrfinder
        if self.lr != self.config.optimizer.params.lr:
            self.config.optimizer.params.lr = self.lr

        new_lr, scaled = self.get_scaled_lr(self.config.optimizer.params.lr)
        if scaled:
            self.config.optimizer.params.lr = new_lr
            rank_zero_print(f"Using scaled LR: {self.config.optimizer.params.lr}")

        params_to_optim = [{'params': self.unet.parameters()}]
        
        if self.config.trainer.get("train_text_encoder") == True:
            text_encoder_group = {'params': self.text_encoder.parameters()}
            if self.config.trainer.get("text_encoder_lr"):
                text_encoder_group['lr'], te_scaled = self.get_scaled_lr(self.config.trainer.get("text_encoder_lr"))
            params_to_optim.append(text_encoder_group)

        optimizer = get_class(self.config.optimizer.name)(
            params_to_optim, **self.config.optimizer.params
        )
        scheduler = get_class(self.config.lr_scheduler.name)(
            optimizer=optimizer,
            **self.config.lr_scheduler.params
        )
        if "transformers" in self.config.lr_scheduler.name:
            scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
            
        return [[optimizer], [scheduler]]

    def on_train_start(self):
        if self.config.get("cast_vae_fp32", True) and hasattr(self, "vae"):
            self.vae.to(torch.float32)
        
        if self.config.trainer.use_ema:
            self.ema.to(self.device, dtype=self.unet.dtype)

        if self.use_latent_cache:
            cache_dir = self.config.dataset.get("cache_dir", "cache")
            self.dataset.update_cache_index(cache_dir, self.local_rank)
            self.trainer.strategy.barrier()

            allclose = self.dataset.cache_latents(self.encode_pixels)
            self.trainer.strategy.barrier()
            if not allclose:
                self.dataset.update_cache_index(cache_dir, self.local_rank)
                    
            # wait for all processes to finish combining the cache
            self.trainer.strategy.barrier()

    def on_train_epoch_start(self) -> None:
        if self.use_latent_cache and hasattr(self, "vae"):
            self.vae.to("cpu")
            
        if self.is_sdxl:
            self.text_encoder.to("cpu")
            self.text_encoder_2.to("cpu")

    def on_train_batch_end(self, *args, **kwargs):
        if self.config.trainer.use_ema:
            self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        if not self.is_sdxl:
            state_dict = convert_to_sd(checkpoint["state_dict"])
            
        if self.config.checkpoint.get("extended"):
            if self.config.checkpoint.extended.save_fp16_weights:
                state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}

        checkpoint["state_dict"] = state_dict

        if self.config.trainer.use_ema:
            checkpoint["model_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.is_sdxl:
            unet_sd, vae_sd, te, te2 = convert_to_df(checkpoint["state_dict"])
            self.unet.load_state_dict(unet_sd)
            self.vae.load_state_dict(vae_sd)
            self.text_encoder.load_state_dict(te.state_dict())
            self.text_encoder_2.load_state_dict(te2.state_dict())
        else:
            unet_sd, vae_sd, te_sd = convert_to_df(checkpoint["state_dict"])
            self.unet.load_state_dict(unet_sd)
            self.vae.load_state_dict(vae_sd)
            self.text_encoder.load_state_dict(te_sd)
            
        checkpoint["state_dict"] = {}
        if self.config.trainer.use_ema:
            self.ema.load_state_dict(checkpoint["model_ema"])
