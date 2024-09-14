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
import torch.nn.functional as F

from common.logging import logger
from common.model_utils import *
from common.utils import log_image


class FluxModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.target_device = device
        self.init_model()

    def init_model(self):
        timer = time.perf_counter()
        
        model_name = self.config.get("model_name", "")
        assert model_name in ["flux-schnell", "flux-dev", "flux-dev-cfg"], f"Invalid model name: {model_name}"
        
        logger.info(f"Initializing model {model_name}")
        self.model, ae, t5, clip = load_models(
            model_name,
            ckpt_path=self.config.get("ckpt_path", "/storage/dev/nyanko/flux-dev/flux1-dev.sft"),
            ae_path=self.config.get("ae_path", "/storage/dev/nyanko/flux-dev/ae.sft"),
            device=self.target_device,
        )
        
        self.use_attention_mask = self.config.get("use_attention_mask", False)
        self.ae, self.t5, self.clip = [ae], [t5], [clip]
        self.vae_encode_bsz = 8
        logger.info("Model initialized in {:.2f}s".format(time.perf_counter() - timer))

    @torch.inference_mode()
    def sample(
        self,
        prompt,
        seed=42,
        size=(1024, 1024),
        steps=28,
    ):
        t5 = self.t5[0] if isinstance(self.t5, list) else self.t5
        clip = self.clip[0] if isinstance(self.clip, list) else self.clip
        ae = self.ae[0] if isinstance(self.ae, list) else self.ae
        
        ae.to(self.target_device)
        height, width = size
        x = get_noise(1, height, width, device=self.target_device, dtype=torch.bfloat16, seed=seed)
        timesteps = get_schedule(steps, (x.shape[-1] * x.shape[-2]) // 4, shift=True)
        inp = prepare(t5=t5, clip=clip, img=x, prompt=prompt)
        if not self.use_attention_mask:
            inp["txt_attn_mask"] = None
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=3)
            x = unpack(x.float(), height, width)
            latents = ae.decode(x)
            
        image = torch.clamp((latents + 1.0) / 2.0, min=0.0, max=1.0).cpu().float()
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]
        return image

    def forward(self, batch):
        t5 = self.t5[0] if isinstance(self.t5, list) else self.t5
        clip = self.clip[0] if isinstance(self.clip, list) else self.clip
        ae = self.ae[0] if isinstance(self.ae, list) else self.ae
        
        with torch.no_grad():
            if "latent" in batch["extras"]:
                latents = torch.stack([t["latent"] for t in batch["extras"]])
            else:
                ae = ae.to(self.target_device)
                x = batch["pixels"].float()
                latents = []
                for i in range(0, x.shape[0], self.vae_encode_bsz):
                    o = x[i : i + self.vae_encode_bsz]
                    latents.append(ae.encode(o))
                latents = torch.cat(latents, dim=0)

            bsz, c, h, w = latents.shape 
            model_dtype = torch.bfloat16 #next(self.model.parameters()).dtype 
            
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # u = torch.normal(mean=0.0, std=1.0, size=(len(latents),), device=self.target_device)
            # t = torch.nn.functional.sigmoid(u)
            t = torch.sigmoid(torch.randn((bsz,), device=self.target_device))

            image_seq_len = h * w // 4
            mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
            t = time_shift(mu, 1.0, t)
            # t = torch.rand((bsz,), device=self.target_device)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            t_ = t.view(t.size(0), *([1] * (len(latents.size()) - 1)))
            noised_latents = (1 - t_) * latents + t_ * noise
            
            forward_args = prepare(t5, clip, noised_latents, batch["prompts"])
            forward_args = {
                "img": forward_args["img"].to(model_dtype),
                "img_ids": forward_args["img_ids"],
                "txt": forward_args["txt"],
                "txt_ids": forward_args["txt_ids"],
                "txt_attn_mask": forward_args["txt_attn_mask"] if self.use_attention_mask else None,
                "y": forward_args["vec"],
                "guidance": torch.tensor([1.0] * bsz, device=self.target_device, dtype=model_dtype),
                "timesteps": t.to(model_dtype)
            }
        
        model_pred = self.model(**forward_args)
        model_pred = unpack(model_pred, h*8, w*8)
        
        target = noise - latents # for loss v
        loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        return loss.mean()

    @torch.inference_mode()
    def generate_samples(self, current_epoch, global_step, world_size=1, rank=0):
        if world_size > 2:
            self.generate_samples_dist(world_size, rank, current_epoch, global_step)
            return dist.barrier()
        # if rank in [0, -1]:
        return self.generate_samples_seq(current_epoch, global_step)

    def generate_samples_dist(self, world_size, rank, current_epoch, global_step):
        config = self.config.sampling
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        prompts = prompts[:world_size]
        local_prompts = prompts[rank::world_size]
        save_dir = Path(config.save_dir)
        for idx, prompt in tqdm(
            enumerate(local_prompts), desc=f"Sampling (Process {rank})", total=len(local_prompts), leave=False
        ):
            image = self.sample(prompt, size=size, seed=config.seed)[0]
            image.save(save_dir / f"sample_e{current_epoch}_s{global_step}_p{rank}_{idx}.png")
        
        self.model.train()
        dist.barrier()
        if rank in [0, -1]:
            all_images = []
            all_prompts = []
            for rank in range(world_size):
                local_prompts = prompts[rank::world_size]
                for idx, prompt in enumerate(local_prompts):
                    img = Image.open(save_dir / f"sample_e{current_epoch}_s{global_step}_p{rank}_{idx}.png")
                    all_prompts.append(prompt)
                    all_images.append(img)

            if config.use_wandb:
                log_image(key="samples", images=all_images, caption=all_prompts, step=global_step)
    
    def generate_samples_seq(self, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))
        self.model.eval()

        for idx, prompt in tqdm(
            enumerate(prompts), desc="Sampling", total=len(prompts), leave=False
        ):
            image = self.sample(prompt, size=size, seed=config.seed)
            image[0].save(
                Path(config.save_dir)
                / f"sample_e{current_epoch}_s{global_step}_{idx}.png"
            )
            images.extend(image)

        self.model.train()
        if config.use_wandb:
            log_image(key="samples", images=images, caption=prompts, step=global_step)
