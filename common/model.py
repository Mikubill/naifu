import copy
import math
from pathlib import Path
import torch
import torch.utils.checkpoint
from PIL import Image

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist

from common.sdvae import AutoEncoder, SD3LatentFormat
from common.mmdit import MMDiT
from common.utils import load_torch_file, EmptyInitWrapper
from common.logging import logger

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTextConfig, T5Config, CLIPTokenizer, T5Tokenizer
from common.utils import log_image

class MMDITWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.diffusion_model = MMDiT(**kwargs)
        
    def forward(self, x, t, c, y) -> torch.Tensor:
        model_dtype = next(self.diffusion_model.parameters()).dtype
        x, c, y = x.to(model_dtype), c.to(model_dtype), y.to(model_dtype)        
        return self.diffusion_model(x, t, context=c, y=y)

def load_file_from_path(path, model, device, extra_keys={}):
    sd = load_torch_file(path, device)
    sd.update(extra_keys)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        logger.info(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        logger.info(f"Unexpected Keys: {unexpected}")

# define the LightningModule
class StableDiffusionModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.target_device = device
        self.init_model()

    def init_model(self):
        advanced = self.config.get("advanced", {})
        logger.info(f"Loading model from {self.config.model_path}")
        sd = load_torch_file(self.config.model_path, self.target_device)

        # build models
        tte_1 = advanced.get("train_text_encoder_1", False)
        tte_2 = advanced.get("train_text_encoder_2", False)
        tte_3 = advanced.get("train_text_encoder_3", False)
        
        # parse params from model sd
        patch_size = sd[f"model.diffusion_model.x_embedder.proj.weight"].shape[2]
        depth = sd[f"model.diffusion_model.x_embedder.proj.weight"].shape[0] // 64
        num_patches = sd[f"model.diffusion_model.pos_embed"].shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = sd[f"model.diffusion_model.y_embedder.mlp.0.weight"].shape[1]
        context_shape = sd[f"model.diffusion_model.context_embedder.weight"].shape
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {"in_features": context_shape[1], "out_features": context_shape[0]}
        }
        logger.info(f"Detected model params: patch_size={patch_size}, depth={depth}, num_patches={num_patches}")
        
        with EmptyInitWrapper():
            self.first_stage_model = AutoEncoder(
                device=self.target_device, 
                # dtype=torch.float16
            )
            
            self.model = MMDITWrapper(
                input_size=None, 
                pos_embed_scaling_factor=None, 
                pos_embed_offset=None, 
                pos_embed_max_size=pos_embed_max_size, 
                patch_size=patch_size, 
                in_channels=16, 
                depth=depth, 
                num_patches=num_patches, 
                adm_in_channels=adm_in_channels,
                context_embedder_config=context_embedder_config, 
                device=self.target_device, 
                # dtype=torch.float16,
                compile_core=False,
                use_checkpoint=True
            )
        
        self.first_stage_model.eval()
        self.first_stage_model.requires_grad_(False)
            
        logger.info(f"Loading mmdit model from {self.config.model_path}")
        missing, unexpected = self.load_state_dict(sd, strict=True)
        if len(missing) > 0:
            logger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")
        
        logger.info(f"Loading tokenizers")
        df_model = "stabilityai/stable-diffusion-3-medium-diffusers"
        self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(df_model, subfolder="tokenizer")
        self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(df_model, subfolder="tokenizer_2")
        self.t5xxl_tokenizer = T5Tokenizer.from_pretrained(df_model, subfolder="tokenizer_3")
            
        with EmptyInitWrapper():
            clip_l_config = CLIPTextConfig.from_pretrained(df_model, subfolder="text_encoder")
            clip_g_config = CLIPTextConfig.from_pretrained(df_model, subfolder="text_encoder_2")
            t5_config = T5Config.from_pretrained(df_model, subfolder="text_encoder_3")            
            self.clip_l = CLIPTextModelWithProjection(clip_l_config)
            self.clip_g = CLIPTextModelWithProjection(clip_g_config)
            self.t5xxl = T5EncoderModel(t5_config)
        
        assert self.config.clip_l_path is not None, "clip_l_path is required"
        logger.info(f"Loading clip_l model from {self.config.clip_l_path}")
        
        # https://github.com/huggingface/diffusers/blob/src/diffusers/loaders/single_file_utils.py#L1389
        position_embedding_dim = self.clip_l.text_model.embeddings.position_embedding.weight.shape[-1]
        extra_keys = {"text_projection.weight": torch.eye(position_embedding_dim)}
        load_file_from_path(self.config.clip_l_path, self.clip_l, self.target_device, extra_keys=extra_keys)
        self.clip_l.requires_grad_(False).to(self.target_device)
        if tte_1: 
            self.clip_l.requires_grad_(True)
            self.clip_l.gradient_checkpointing_enable()
        
        assert self.config.clip_g_path is not None, "clip_g_path is required"
        logger.info(f"Loading clip_g model from {self.config.clip_g_path}")
        load_file_from_path(self.config.clip_g_path, self.clip_g, self.target_device)
        self.clip_g.requires_grad_(False).to(self.target_device)
        if tte_2: 
            self.clip_g.requires_grad_(True)
            self.clip_g.gradient_checkpointing_enable()
        
        if self.config.t5xxl_path is not None:
            logger.info(f"Loading t5xxl model from {self.config.t5xxl_path}")
            load_file_from_path(self.config.t5xxl_path, self.t5xxl, self.target_device)
            self.t5xxl.requires_grad_(False).to(self.target_device)
            if tte_3: 
                self.t5xxl.requires_grad_(True)
                self.t5xxl.gradient_checkpointing_enable()
        else:
            self.t5xxl = None

        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = advanced.get("vae_encode_batch_size", self.batch_size)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = SD3LatentFormat().process_out(z)
        out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        latents = []
        for i in range(0, x.shape[0], self.vae_encode_bsz):
            o = x[i : i + self.vae_encode_bsz]
            latents.append(self.first_stage_model.encode(o))
        z = torch.cat(latents, dim=0)
        return SD3LatentFormat().process_in(z)
    
    def _clip_encode_prompt(self, prompt, tokenizer, model):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = model(text_input_ids.to(self.target_device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        return prompt_embeds, pooled_prompt_embeds
    
    def encode_prompt(self, prompt: str):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_l_prompt_embeds, clip_l_pooled_prompt_embeds = self._clip_encode_prompt(
            prompt, self.clip_l_tokenizer, self.clip_l
        )
        clip_g_prompt_embeds, clip_g_pooled_prompt_embeds = self._clip_encode_prompt(
            prompt, self.clip_g_tokenizer, self.clip_g
        )
        clip_pooled_prompt_embeds_list = [clip_l_pooled_prompt_embeds, clip_g_pooled_prompt_embeds]
        clip_prompt_embeds_list = [clip_l_prompt_embeds, clip_g_prompt_embeds]

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_text_inputs = self.t5xxl_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        t5_text_input_ids = t5_text_inputs.input_ids
        if self.t5xxl is not None:
            t5_prompt_embeds = self.t5xxl(t5_text_input_ids.to(self.target_device))[0]
        else:
            t5_prompt_embeds = torch.zeros(
                t5_text_input_ids.shape[0], 77, 4096, dtype=torch.float32, device=self.target_device
            )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
        return prompt_embeds, pooled_prompt_embeds

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
        noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        cond, cond_pooled = self.encode_prompt(prompt=prompt)
        uncond, uncond_pooled = self.encode_prompt(prompt=negative_prompt)
        
        cond_in = torch.cat([uncond, cond], dim=0).to(self.target_device)
        cond_pooled_in = torch.cat([uncond_pooled, cond_pooled], dim=0).to(self.target_device)

        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)
        size = (height, width)
        latents_shape = (1, 16, size[0] // 8, size[1] // 8)
        latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = latents.to(self.target_device)
        
        noise_scheduler_copy.set_timesteps(steps)
        timesteps = noise_scheduler_copy.timesteps
        num_latent_input = 2
        
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = latent_model_input.to(self.target_device)
            
            timestep = t.expand(latent_model_input.shape[0]).to(self.target_device)
            noise_pred = self.model(latent_model_input, timestep, cond_in, cond_pooled_in)
            pred_uncond, pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
            noise_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler_copy.step(noise_pred, t, latents).prev_sample

        latents = self.decode_first_stage(latents.to(torch.float32))
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image
    
    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.target_device)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device=self.target_device)
        timesteps = timesteps.to(device=self.target_device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["pixels"].float())
        else:
            self.first_stage_model.cpu()
            latents = SD3LatentFormat().process_in(batch["pixels"])
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=torch.float32)
        bsz = latents.shape[0]
        
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        weighting_scheme = advanced.get("weighting_scheme", "logit_normal")
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            logit_mean, logit_std = advanced.get("logit_mean", 0.0), advanced.get("logit_std", 1.0)
            u = torch.normal(mean=logit_mean, std=logit_std, size=(bsz,), device=self.target_device)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            mode_scale = advanced.get("mode_scale", 1.29)
            u = torch.rand(size=(bsz,), device=self.target_device)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(bsz,), device=self.target_device)

        # Sample a random timestep for each image
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long().cpu()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.target_device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # Predict the noise residual
        prompt_embeds, pooled_prompt_embeds  = self.encode_prompt(batch["prompts"])
        noise_pred = self.model(noisy_model_input, timesteps, prompt_embeds, pooled_prompt_embeds)

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = noise_pred * (-sigmas) + noisy_model_input

        # Compute the loss
        weighting = torch.ones_like(sigmas)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        elif weighting_scheme == "cosmap":
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)

        # simplified flow matching aka 0-rectified flow matching loss
        # target = model_input - noise
        target = latents
        
        # Compute regular loss.
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3]) * weighting
        loss = loss.mean()  # mean over batch dimension
        return loss

    def generate_samples(self, current_epoch, global_step, world_size=1, rank=0):
        if world_size > 2:
            self.generate_samples_dist(world_size, rank, current_epoch, global_step)
            return dist.barrier()
                
        return self.generate_samples_seq(current_epoch, global_step)

    def generate_samples_dist(self, world_size, rank, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()
        self.first_stage_model.to(self.target_device, dtype=torch.float32)

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
        self.first_stage_model.to(self.target_device)

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
