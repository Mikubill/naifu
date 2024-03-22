import math
import torch
import torchvision
import torch.utils.checkpoint
import numpy as np
import lightning as pl

from PIL import Image
from tqdm import tqdm 
from common.utils import load_torch_file, EmptyInitWrapper
from lightning.pytorch.utilities import rank_zero_only

from pathlib import Path
from safetensors.torch import save_file
from models.cascade.effnet import EfficientNetEncoder
from models.cascade.stage_a import StageA
from models.cascade.stage_b import StageB
from models.cascade.stage_c import StageC
from models.cascade.previewer import Previewer
from models.gdf import GDF, CosineSchedule, VPScaler, EpsilonTarget, CosineTNoiseCond, AdaptiveLossWeight, P2LossWeight
from transformers import AutoTokenizer, CLIPTextModelWithProjection

import logging
logger = logging.getLogger("Trainer")

CLIP_TEXT_MODEL_NAME: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
EFFNET_PREPROCESS = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    )
])

# https://github.com/Stability-AI/StableCascade/blob/7a7d341f729ccaa042920a1fac3e7b9326079aca/inference/utils.py#L55
def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = math.ceil(height / compression_factor_b)
    latent_width = math.ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)

    latent_height = math.ceil(height / compression_factor_a)
    latent_width = math.ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)

    return stage_c_latent_shape, stage_b_latent_shape

class StableCascadeModel(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.target_device = device
        self.init_model()

    def build_models(self, stage_a=False, stage_b=False, stage_c=True):
        config = self.config

        with EmptyInitWrapper(self.target_device):
            self.effnet = EfficientNetEncoder()
            effnet_sd = load_torch_file(self.model_path.effnet, self.target_device)
            self.effnet.load_state_dict(effnet_sd)
            
            if stage_a:
                self.stage_a = StageA()
                stage_a_sd = load_torch_file(self.model_path.stage_a, self.target_device)
                self.stage_a.load_state_dict(stage_a_sd)

            if stage_b:
                self.stage_b = StageB()
                stage_b_sd = load_torch_file(self.model_path.stage_b, self.target_device)
                self.stage_b.load_state_dict(stage_b_sd)

            if stage_c:
                self.stage_c = StageC()
                stage_c_sd = load_torch_file(self.model_path.stage_c, self.target_device)
                self.stage_c.load_state_dict(stage_c_sd)

            self.previewer = Previewer()
            previewer_sd = load_torch_file(self.model_path.previewer, self.target_device)
            self.previewer.load_state_dict(previewer_sd)
        
        clip_text_model_name = self.config.get("clip_text_model_name", CLIP_TEXT_MODEL_NAME)        
        self.tokenizer = AutoTokenizer.from_pretrained(clip_text_model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_text_model_name)
        
        self.gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(),
            target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight else P2LossWeight(),
        )
        self.to(self.target_device)
        self.previewer.requires_grad_(False).eval()
        self.effnet.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()
        
    def init_model(self):
        advanced = self.config.get("advanced", {})
        self.build_models()
        self.stage_c.set_gradient_checkpointing(True)
        if advanced.get("train_text_encoder"):
            self.text_encoder.requires_grad_(True)
            self.text_encoder.gradient_checkpointing_enable()
        
        self.max_token_length = self.config.dataset.get("max_token_length", 75) + 2
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

    @torch.no_grad() 
    def encode_pixels(self, x):
        latents = []
        with torch.autocast("cuda", enabled=False):
            for i in range(0, x.shape[0], self.vae_encode_bsz):
                o = x[i : i + self.vae_encode_bsz]
                latents.append(self.effnet(o))
        z = torch.cat(latents, dim=0)
        return z

    def encode_prompts(self, prompts):            
        input_ids = self.tokenizer(
            prompts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_token_length,
            return_tensors="pt"
        ).input_ids 
        tokenizer_max_length = self.tokenizer.model_max_length
        oids = []
        
        for iids in input_ids:
            z = []
            for i in range(1, self.max_token_length - tokenizer_max_length + 2, tokenizer_max_length - 2):  # (1, 152, 75)
                ids_chunk = (
                    iids[0].unsqueeze(0),
                    iids[i : i + tokenizer_max_length - 2],
                    iids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                z.append(ids_chunk)
            oids.append(torch.stack(z))
            
        oids = torch.stack(oids)
        
        # b,n,77 -> b*n, 77
        bs = oids.size(0)
        input_ids = oids.reshape((-1, tokenizer_max_length))
        
        # text_encoder
        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        state = self.text_encoder(input_ids.to(self.device), output_hidden_states=True)
        pooled_states = state["text_embeds"]
        hidden_states = state["hidden_states"][-1]
        encoder_hidden_states = hidden_states.reshape((bs, -1, hidden_states.shape[-1]))
        
        group_size = 1 if self.max_token_length is None else self.max_token_length // 75
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, self.max_token_length, tokenizer_max_length):
            states_list.append(encoder_hidden_states[:, i : i + tokenizer_max_length - 2])  
            
        states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
        encoder_hidden_states = torch.cat(states_list, dim=1)
        pooled_states = pooled_states[::group_size]
        return encoder_hidden_states, pooled_states

    @rank_zero_only
    def generate_samples(self, logger, current_epoch, global_step):
        config = self.config.sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        images = []
        size = (config.get("height", 1024), config.get("width", 1024))

        for idx, prompt in tqdm(enumerate(prompts), desc="Sampling", leave=False):
            image = self.sample(prompt, size=size, generator=generator)
            image[0].save(Path(config.save_dir) / f"sample_e{current_epoch}_s{global_step}_{idx}.png")
            images.extend(image)

        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(key="samples", images=images, caption=prompts, step=global_step)
            
    @torch.inference_mode()
    def sample(
        self,
        prompt,
        negative_prompt="lowres, low quality, text, error, extra digit, cropped",
        generator=None,
        size=(1024, 1024),
        steps=20,
        guidance_scale=5,
    ):
        height, width = size
        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8) 
        size = (height, width)
        
        hidden, pool = self.encode_prompts(prompt)
        uncond_hidden, uncond_pool = self.encode_prompts(negative_prompt)
        pool = pool.unsqueeze(1)
        uncond_pool = uncond_pool.unsqueeze(1)
        
        stage_c_latent_shape = calculate_latent_sizes(height, width, batch_size=1)[0]
        init_latents = torch.randn(stage_c_latent_shape, generator=generator).to(self.target_device)
        image_embed = torch.zeros(1, 768, device=self.target_device)
        conditions = {
            "clip_text_pooled": pool, 
            "clip_text": hidden, 
            "clip_img": image_embed
        }
        unconditions = {
            "clip_text_pooled": uncond_pool, 
            "clip_text": uncond_hidden, 
            "clip_img": image_embed
        }
        sampling_c = self.gdf.sample(
            self.stage_c,
            conditions,
            stage_c_latent_shape,
            unconditions,
            x_init=init_latents,
            device=self.target_device,
            cfg=guidance_scale,
            shift=2,
            timesteps=steps,
            t_start=1.0,
        )
        for sampled_c, _, _ in tqdm(sampling_c, total=steps):
            sampled_c = sampled_c
        
        previewer = self.previewer
        sampled_c = sampled_c.to(self.target_device)
        image = torch.clamp(previewer(sampled_c)[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
        image = Image.fromarray((image * 255).astype(np.uint8))
        return [image]
    
    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        cfg = self.config.trainer
        if self.config.advanced.get("train_text_encoder"):
            self.text_encoder.save_pretrained(f"{model_path}_text_encoder")

        state_dict = self.stage_c.state_dict()
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

    def forward(self, batch):
        raise NotImplementedError

