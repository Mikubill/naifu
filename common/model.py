import math
import torch
import torch.utils.checkpoint
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import time
import torch.distributed as dist

from common.hydit import DiTModel, MT5Embedder
from common.utils import get_class, load_torch_file, EmptyInitWrapper
from common.logging import logger

from diffusers import DDPMScheduler, AutoencoderKL
from transformers import AutoTokenizer, BertModel
from common.utils import log_image
from common.model_utils import cache_snr_values, calc_snr_weight, apply_zero_terminal_snr
from common.model_utils import clip_process_input_ids, clip_get_hidden_states, calc_rope

class StableDiffusionModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.target_device = device
        self.init_model()

    def init_model(self):
        advanced = self.config.get("advanced", {})
        sd = load_torch_file(self.config.model_path, self.target_device)
        timer = time.perf_counter()

        # build models
        tte_1 = advanced.get("train_text_encoder_1", False)
        tte_2 = advanced.get("train_text_encoder_2", False)
        
        logger.info("Building dit model")
        with EmptyInitWrapper():
            # use_extra_cond=True,  # for hydit 1.1
            # use_extra_cond=False, # for hydit 1.2
            use_extra_cond = self.config.get("version", "hydit_1.2") == "hydit_1.1"
            self.use_extra_cond = use_extra_cond
            self.beta_end = 0.03 if use_extra_cond else 0.018
            self.model = DiTModel(
                input_size=(1024//8, 1024//8),
                use_extra_cond=use_extra_cond,
                depth=40, hidden_size=1408, patch_size=2, num_heads=16, mlp_ratio=4.3637,  # hydit default params for dig_g_2
            )
            self.model.enable_gradient_checkpointing()

        hf_model = "Tencent-Hunyuan/HunyuanDiT-v1.2"
        logger.info(f"Loading vae model from {hf_model}")
        
        self.vae = AutoencoderKL.from_pretrained(hf_model, subfolder="t2i/sdxl-vae-fp16-fix")
        self.vae.eval()
        self.vae.requires_grad_(False)

        logger.info(f"Loading mmdit model from {self.config.model_path}")
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        if len(missing) > 0:
            logger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

        logger.info(f"Loading clip model from {hf_model}")
        self.clip_tokenizer = AutoTokenizer.from_pretrained(hf_model, subfolder="t2i/tokenizer")
        self.clip_tokenizer.eos_token_id = 2
        # assert self.clip_tokenizer.eos_token_id == 2, f"eos_token_id: {self.clip_tokenizer.eos_token_id}"
        
        clip_encoder = BertModel.from_pretrained(hf_model, subfolder="t2i/clip_text_encoder")
        if self.config.trainer.get("clip_path", None):
            logger.info(f"Loading clip model from {self.config.trainer.clip_path}")
            clip_encoder.load_state_dict(load_torch_file(self.config.trainer.clip_path, self.target_device))

        logger.info(f"Loading mt5 model from {hf_model}")
        mt5_embedder = MT5Embedder(hf_model, model_kwargs=dict(subfolder="t2i/mt5"), torch_dtype=torch.float16, max_length=256)
        if self.config.trainer.get("mt5_path", None):
            logger.info(f"Loading mt5 model from {self.config.trainer.mt5_path}")
            mt5_embedder.load_state_dict(load_torch_file(self.config.trainer.mt5_path, self.target_device))

        self.model.train()
        self.clip_encoder = [clip_encoder.to(self.target_device).requires_grad_(False)]
        self.mt5_embedder = [mt5_embedder.to(self.target_device).requires_grad_(False)]

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=self.beta_end,
            beta_schedule="scaled_linear", num_train_timesteps=1000,
            clip_sample=False, steps_offset=1,
        )
        self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(self.target_device)
        
        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)
            
        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)
            
        self.scale_factor = 0.13025
        self.vae_encode_bsz = 8  
        logger.info("Model initialized in {:.2f}s".format(time.perf_counter() - timer))

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        out = self.vae.decode(z, return_dict=False)[0]
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        latents = []
        for i in range(0, x.shape[0], self.vae_encode_bsz):
            o = x[i : i + self.vae_encode_bsz]
            latents.append(self.vae.encode(o).latent_dist.sample())
        z = torch.cat(latents, dim=0)
        return self.scale_factor * z
    
    @torch.no_grad()
    def encode_cond(self, latents, batch, dtype=torch.float16):
        clip_encoder = self.clip_encoder[0] if isinstance(self.clip_encoder, list) else self.clip_encoder
        mt5_embedder = self.mt5_embedder[0] if isinstance(self.mt5_embedder, list) else self.mt5_embedder
        
        prompts = batch["prompts"]        
        max_length_clip = 75 + 2
        clip_tokens = self.clip_tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length_clip,
            return_tensors="pt",
        )
        ts = clip_tokens["input_ids"]
        input_ids = torch.stack([clip_process_input_ids(t, self.clip_tokenizer, max_length_clip) for t in ts])
        # clip_input_ids = clip_process_input_ids(clip_tokens, self.clip_tokenizer, max_length_clip)
        clip_hidden_states, clip_mask = clip_get_hidden_states(
            input_ids.to(self.target_device),
            self.clip_tokenizer,
            clip_encoder,
            max_token_length=max_length_clip,
        )

        # mt5ids, mt5mask = self.mt5_embedder.get_tokens_and_mask(prompts)
        mt5_hidden_states, mt5_mask = mt5_embedder.get_text_embeddings(prompts)
        bsz, c, h, w = latents.shape
    
        cos_cis_img, sin_cis_img = calc_rope(h * 8, w * 8, 2, 88)
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        
        style = None
        image_meta_size = None
        if self.use_extra_cond:
            style = torch.as_tensor([0] * bsz, device=self.target_device)
            image_meta_size = torch.concat([orig_size, target_size, crop_size]).to(self.target_device)
        
        cond = dict(
            encoder_hidden_states=clip_hidden_states.to(dtype),
            text_embedding_mask=clip_mask.long().to(self.target_device),
            encoder_hidden_states_t5=mt5_hidden_states.to(dtype),
            text_embedding_mask_t5=mt5_mask.long().to(self.target_device),
            cos_cis_img=cos_cis_img.to(self.target_device),
            sin_cis_img=sin_cis_img.to(self.target_device),
            style=style,
            image_meta_size=image_meta_size,
        )
        return cond

    @torch.inference_mode()
    def sample(
        self,
        prompt,
        negative_prompt="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，",
        generator=None,
        size=(1024, 1024),
        steps=28,
        guidance_scale=6.,
    ):
        self.vae.to(self.target_device)
        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)
        size = (height, width)
        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32).to(self.target_device)
        num_latent_input = 2
        scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=self.beta_end,
            beta_schedule="scaled_linear", num_train_timesteps=1000,
            clip_sample=False, steps_offset=1, prediction_type = "v_prediction",
        )

        prompts_batch = {
            "target_sizes_hw": torch.stack([torch.asarray(size), torch.asarray(size)]).cuda(),
            "original_sizes_hw": torch.stack([torch.asarray(size), torch.asarray(size)]).cuda(),
            "crop_top_lefts": torch.stack([torch.asarray((0, 0)), torch.asarray((0, 0))]).cuda(),
        }
        prompts_batch["prompts"] = [negative_prompt, prompt]
        cond_args = self.encode_cond(
            latents=latents.repeat((num_latent_input, 1, 1, 1)),
            batch=prompts_batch
        )        
        
        with torch.autocast("cuda", dtype=torch.float16):
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(28)
            timesteps = scheduler.timesteps
            num_latent_input = 2
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(self.target_device)

                tond =  torch.asarray([t] * 2).to(self.target_device).to(torch.float16)
                noise_pred = self.model(latent_model_input, tond, **cond_args)
                noise_pred = noise_pred.chunk(2, dim=1)[0]    
                pred_uncond, pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents.to(self.target_device)).prev_sample
                
        latents = self.decode_first_stage(latents.to(torch.float32))
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image

    def forward(self, batch):
        
        with torch.no_grad():
            advanced = self.config.get("advanced", {})
            if not batch["is_latent"]:
                if "latent" in batch["extras"]:
                    latents = torch.stack([t["latent"] for t in batch["extras"]])
                else:
                    self.vae.to(self.target_device)
                    latents = self.encode_first_stage(batch["pixels"].to(self.vae.dtype))
            else:
                self.vae.cpu()
                latents = self.scale_factor * batch["pixels"]

            bsz = latents.shape[0]        
            cond = self.encode_cond(latents, batch)
            model_dtype = next(self.model.parameters()).dtype

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents, dtype=model_dtype)
            timesteps = torch.randint(low=0, high=1000, size=(bsz,), dtype=torch.int64, device=latents.device).long() 
            
            snr_w = 1.0
            min_snr_gamma = advanced.get("min_snr", False)     
            if min_snr_gamma:
                snr_w = calc_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, True)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.model(noisy_latents.to(torch.float16), timesteps, **cond).chunk(2, dim=1)[0]
        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            
        # do not mean over batch dimension for snr weight or scale v-pred loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3]) 

        loss = loss * snr_w        
        loss = loss.mean()  # mean over batch dimension
        return loss

    def generate_samples(self, current_epoch, global_step, world_size=1, rank=0):
        if world_size > 2:
            self.generate_samples_dist(world_size, rank, current_epoch, global_step)
            return dist.barrier()
        # if rank in [0, -1]:
        return self.generate_samples_seq(current_epoch, global_step)

    def generate_samples_dist(self, world_size, rank, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()
        self.vae.to(self.target_device, dtype=torch.float32)

        prompts = prompts[:world_size]
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

            if config.use_wandb:
                log_image(key="samples", images=all_images, caption=all_prompts, step=global_step)
    
    def generate_samples_seq(self, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()
        self.vae.to(self.target_device)

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
        if config.use_wandb:
            log_image(key="samples", images=images, caption=prompts, step=global_step)