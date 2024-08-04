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

@torch.no_grad()
def immiscible_diffusion(self, latents):
    import scipy
    # "Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment" (2024) Li et al. arxiv.org/abs/2406.12303
    noise = torch.randn_like(latents, device=latents.device)
    dist = torch.cdist(latents.view(latents.shape[0], -1), noise.view(noise.shape[0], -1), p=2)
    assign_row, assign_col = scipy.optimize.linear_sum_assignment(dist.cpu().numpy())
    return noise[assign_col]
    
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


class DiscreteVDDPMDenoiser:
    """A wrapper for discrete schedule DDPM models that output v."""

    def __init__(self, model, beta_start=0.00085, beta_end=0.018, num_train_timesteps=1000, device="cuda"):
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.inner_model = model
        self.sigma_data = 1.
        
    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_v(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)
    
    def sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)
    
    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def denoise(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


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

        logger.info("Loading vae model from Tencent-Hunyuan/HunyuanDiT-v1.2")
        self.vae = AutoencoderKL.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2", subfolder="t2i/sdxl-vae-fp16-fix")
        self.vae.eval()
        self.vae.requires_grad_(False)

        logger.info(f"Loading mmdit model from {self.config.model_path}")
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        if len(missing) > 0:
            logger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

        hf_model = "Tencent-Hunyuan/HunyuanDiT-v1.1"
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
        clip_encoder = clip_encoder.to(self.target_device)
        if tte_1:
            clip_encoder.requires_grad_(True)
            self.clip_encoder = clip_encoder
        else:
            clip_encoder.requires_grad_(False)
            self.clip_encoder = [clip_encoder]
            
        mt5_embedder = mt5_embedder.to(self.target_device)
        if tte_2:
            mt5_embedder = mt5_embedder.float()
            mt5_embedder.requires_grad_(True)
            mt5_embedder.gradient_checkpointing_enable()
            self.mt5_embedder = mt5_embedder
        else:
            mt5_embedder.requires_grad_(False)
            self.mt5_embedder = [mt5_embedder]

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=self.beta_end,
            beta_schedule="scaled_linear", num_train_timesteps=1000,
            clip_sample=False, steps_offset=1,
        )
        self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(self.target_device)
        
        # debug only: fast test
        # t = ["aaa1fad0f80ajf298f4", "sdfaj01r3809fjds0afd"]
        # tt = ["sdfaj01r3809fjds0afd", "aaa1fad0f80ajf298f4"]
        # tokens = self.clip_tokenizer(t, padding="max_length", truncation=True, return_tensors="pt").to(self.target_device)
        # tokens2 = self.clip_tokenizer(tt, padding="max_length", truncation=True, return_tensors="pt").to(self.target_device)
        # assert torch.allclose(tokens["input_ids"][0], tokens2["input_ids"][1]), f"debug:diff:clip: {tokens['input_ids']}, {tokens2['input_ids']}"
        # hd = clip_encoder(**tokens).last_hidden_state
        # hd2 = clip_encoder(**tokens2).last_hidden_state
        # assert torch.allclose(hd[0], hd2[1], atol=1e-5), f"debug:diff:clip: {hd[0]}, {hd2[1]}"
        
        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)
            
        if hasattr(self.noise_scheduler, "alphas_cumprod"):
            cache_snr_values(self.noise_scheduler, self.target_device)
            
        self.scale_factor = 0.13025
        self.vae_encode_bsz = 8
        self.kdfloss = self.config.get("use_kdiffusion_loss", False)
        self.denoiser = DiscreteVDDPMDenoiser(
            lambda *args, **kwargs: self.model(*args, **kwargs).chunk(2, dim=1)[0],
            beta_start=0.00085, beta_end=self.beta_end, num_train_timesteps=1000, device=self.target_device,
        )
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
    def encode_cond(self, latents, batch):
        self.clip_encoder = self.clip_encoder[0] if isinstance(self.clip_encoder, list) else self.clip_encoder
        self.mt5_embedder = self.mt5_embedder[0] if isinstance(self.mt5_embedder, list) else self.mt5_embedder
        self.clip_encoder.to(self.target_device)
        self.mt5_embedder.to(self.target_device)
        
        prompts = batch["prompts"]        
        max_length_clip = 225 + 2
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
            self.clip_encoder,
            max_token_length=max_length_clip,
        )
        
        # mt5ids, mt5mask = self.mt5_embedder.get_tokens_and_mask(prompts)
        mt5_hidden_states, mt5_mask = self.mt5_embedder.get_text_embeddings(prompts)
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
            encoder_hidden_states=clip_hidden_states.to(latents.dtype),
            text_embedding_mask=clip_mask.long().to(self.target_device),
            encoder_hidden_states_t5=mt5_hidden_states.to(latents.dtype),
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
        negative_prompt="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺",
        generator=None,
        size=(1024, 1024),
        steps=28,
        guidance_scale=6,
    ):
        self.vae.to(self.target_device)
        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)
        size = (height, width)
        latents_shape = (1, 4, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32).to(self.target_device)
        num_latent_input = 2

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
        def cfg_denoise_func(x, sigma):
            uncond, cond = self.denoiser.denoise(x.repeat(2, 1, 1, 1), sigma.repeat(2), **cond_args).chunk(2, dim=0)
            return uncond + (cond - uncond) * guidance_scale
        
        sigmas = get_sigmas_exponential(steps, self.denoiser.sigma_min, self.denoiser.sigma_max, self.target_device)
        sample = sample_euler(cfg_denoise_func, latents * sigmas[0], sigmas)

        latents = self.decode_first_stage(sample.to(torch.float32))
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image

    def forward(self, batch):
        
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

        if not self.kdfloss:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = self.model(noisy_latents, timesteps, **cond).chunk(2, dim=1)[0]
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            
            # do not mean over batch dimension for snr weight or scale v-pred loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3]) 
        else:
            sigmas = self.denoiser.t_to_sigma(timesteps)
            loss = self.denoiser.loss(latents, noise, sigmas, **cond)

        min_snr_gamma = advanced.get("min_snr", False)     
        if min_snr_gamma:
            snr_w = calc_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, True)
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
