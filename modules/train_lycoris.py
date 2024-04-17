import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import (
    get_latest_checkpoint,
    load_torch_file,
    get_class,
    EmptyInitWrapper,
)
from common.logging import logger
from modules.train_sdxl import SupervisedFineTune
from modules.sdxl_utils import (
    get_hidden_states_sdxl,
    convert_sdxl_text_encoder_2_checkpoint,
)
from modules.sdxl_utils import get_size_embeddings
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
from safetensors.torch import save_file

from diffusers import DDPMScheduler
from transformers import (
    CLIPTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from contextlib import nullcontext

try:
    import logging
    logging.getLogger("LyCORIS").addHandler(logger.handlers[0])
    
    from lycoris import create_lycoris, LycorisNetwork
except ImportError as e:
    raise ImportError(
        f"\n\nError import lycoris: {e} \nTry install lycoris using `pip install lycoris_lora toml`"
    )


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = StableDiffusionModel(
        model_path=model_path, config=config, device=fabric.device
    )
    model.prepare_context(fabric)
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()

    params_to_optim = [{"params": model.lycoris_unet.parameters()}]
    # params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        if hasattr(config.optimizer.params, 'lr'):
            lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
            params_to_optim.append({"params": model.lycoris_te1.parameters(), "lr": lr})
        else:
            params_to_optim.append({"params": model.lycoris_te1.parameters()})

    if config.advanced.get("train_text_encoder_2"):
        if hasattr(config.optimizer.params, 'lr'):
            lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
            params_to_optim.append({"params": model.lycoris_te2.parameters(), "lr": lr})
        else:
            params_to_optim.append({"params": model.lycoris_te2.parameters()})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
        
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_checkpoint(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)

    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    model.lycoris_unet, optimizer = fabric.setup(model.lycoris_unet, optimizer)
    if config.advanced.get("train_text_encoder_1"):
        model.lycoris_te1 = fabric.setup(model.lycoris_te1)
    if config.advanced.get("train_text_encoder_2"):
        model.lycoris_te2 = fabric.setup(model.lycoris_te2)
    
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


def init_text_encoder():
    text_model_1_cfg = CLIPTextConfig.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder"
    )
    text_model_2_cfg = CLIPTextConfig.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2"
    )

    with EmptyInitWrapper():
        text_model_1 = CLIPTextModel._from_config(text_model_1_cfg)
        text_model_2 = CLIPTextModelWithProjection._from_config(text_model_2_cfg)

    tokenizer1 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
    )
    tokenizer2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
    )
    text_model_1 = text_model_1.requires_grad_(False)
    text_model_2 = text_model_2.requires_grad_(False)
    return text_model_1, text_model_2, tokenizer1, tokenizer2


# define the LightningModule
class StableDiffusionModel(SupervisedFineTune):
    def forward(self, batch):
        with self.forward_context:
            return super().forward(batch)
        
    def generate_samples(self, **kwargs):
        with self.forward_context:
            return super().generate_samples(**kwargs)
        
    def prepare_context(self, fabric):
        self.forward_context = nullcontext()
        is_fp16_scaled = self.config.get("_scaled_fp16_precision", False)
        if self.config.lightning.precision == "16-true" or is_fp16_scaled:
            self.forward_context = fabric.autocast()
            self.model.to(torch.float16)
        elif self.config.lightning.precision == "bf16-true":
            self.forward_context = fabric.autocast()
            self.model.to(torch.bfloat16)
    
    def init_model(self):
        advanced = self.config.get("advanced", {})
        sd = load_torch_file(self.model_path, self.target_device)
        vae, unet, _ = self.build_models(init_conditioner=False)
        self.first_stage_model = vae
        self.model = unet

        te1_sd, te2_sd, unet_sd, vae_sd = {}, {}, {}, {}
        for k in list(sd.keys()):
            if k.startswith("conditioner.embedders.0.transformer."):
                target = k.replace("conditioner.embedders.0.transformer.", "")
                te1_sd[target] = sd.pop(k)
            elif k.startswith("conditioner.embedders.1.model."):
                target = k
                te2_sd[target] = sd.pop(k)
            elif k.startswith("first_stage_model."):
                target = k.replace("first_stage_model.", "")
                vae_sd[target] = sd.pop(k)
            elif k.startswith("model.diffusion_model."):
                target = k.replace("model.diffusion_model.", "diffusion_model.")
                unet_sd[target] = sd.pop(k)

        converted_sd = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
        self.text_encoder_1, self.text_encoder_2, self.tokenizer_1, self.tokenizer_2 = init_text_encoder()
        self.text_encoder_1.to(self.target_device)
        self.text_encoder_2.to(self.target_device)
        self.text_encoder_1.load_state_dict(te1_sd, strict=False)
        self.text_encoder_2.load_state_dict(converted_sd, strict=False)
        vae.load_state_dict(vae_sd)
        unet.load_state_dict(unet_sd)
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        self.batch_size = self.config.trainer.batch_size
        self.vae_encode_bsz = self.config.get("vae_encode_batch_size", self.batch_size)
        if self.vae_encode_bsz < 0:
            self.vae_encode_bsz = self.batch_size

        if advanced.get("zero_terminal_snr", False):
            apply_zero_terminal_snr(self.noise_scheduler)
        cache_snr_values(self.noise_scheduler, self.target_device)
        
        self.model.diffusion_model.train()
        self.text_encoder_1.train()
        self.text_encoder_2.train()
        self.init_lycoris()

    def init_lycoris(self):
        cfg = self.config
        default_cfg = cfg.lycoris
        self.lycoris_mapping = lycoris_mapping = {}
        self.model.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        LycorisNetwork.apply_preset({"target_name": ".*"})

        logger.info("")
        logger.info(f"Initializing model.lycoris_unet with {cfg.lycoris}")
        self.lycoris_unet = create_lycoris(self.model.diffusion_model, **cfg.get("lycoris_unet", default_cfg))
        
        logger.info("")
        logger.info(f"Initializing model.lycoris_te1 with {cfg.lycoris}")
        self.lycoris_te1 = create_lycoris(self.text_encoder_1, **cfg.get("lycoris_te1", default_cfg))
        
        logger.info("")
        logger.info(f"Initializing model.lycoris_te2 with {cfg.lycoris}")
        self.lycoris_te2 = create_lycoris(self.text_encoder_2, **cfg.get("lycoris_te2", default_cfg))
        lycoris_mapping["unet"] = self.lycoris_unet
        lycoris_mapping["te1"] = self.lycoris_te1
        lycoris_mapping["te2"] = self.lycoris_te2

        self.lycoris_unet.to(self.target_device).apply_to()
        self.lycoris_unet.requires_grad_(True)

        if self.config.advanced.get("train_text_encoder_1"):
            self.lycoris_te1.to(self.target_device).apply_to()
            self.lycoris_te1.requires_grad_(True)

        if self.config.advanced.get("train_text_encoder_2"):
            self.lycoris_te2.to(self.target_device).apply_to()
            self.lycoris_te2.requires_grad_(True)
            
        self.text_encoder_1.text_model.embeddings.requires_grad_(True)
        self.text_encoder_2.text_model.embeddings.requires_grad_(True) 
        
    def get_module(self):
        return self.lycoris_unet       

    def encode_batch(self, batch):
        hidden1, hidden2, pooled = get_hidden_states_sdxl(
            batch["prompts"],
            self.max_token_length,
            self.tokenizer_1,
            self.tokenizer_2,
            self.text_encoder_1,
            self.text_encoder_2,
        )
        emb = get_size_embeddings(
            batch["target_size_as_tuple"],
            batch["original_size_as_tuple"],
            batch["crop_coords_top_left"],
            self.target_device,
        )
        cond = {
            "crossattn": torch.cat([hidden1, hidden2], dim=2),
            "vector": torch.cat([pooled, emb], dim=1),
        }
        return cond

    def load_checkpoint(self, sd):
        sd = sd["state_dict"] if "state_dict" in sd else sd
        te, te2, unet = {}, {}, {}
        for key in list(sd.keys()):
            if key.startswith("lora_unet"):
                unet[key.replace("lora_unet_", "lycoris_")] = sd.pop(key)
            elif key.startswith("lora_te1"):
                te[key.replace("lora_te1_", "lycoris_")] = sd.pop(key)
            elif key.startswith("lora_te2"):
                te2[key.replace("lora_te2_", "lycoris_")] = sd.pop(key)

        self.lycoris_unet.load_state_dict(unet)
        if self.config.advanced.get("train_text_encoder_1"):
            self.lycoris_te1.load_state_dict(te)

        if self.config.advanced.get("train_text_encoder_2"):
            self.lycoris_te2.load_state_dict(te2)

    @rank_zero_only
    def save_checkpoint(self, model_path, metadata):
        cfg = self.config.trainer
        state_dict = {}

        # build lycoris state_dict
        for key, module in self.lycoris_mapping.items():
            module_state_dict = module.state_dict()
            new_state_dict = {}
            for k, v in module_state_dict.items():
                k = k.replace("module.", "")
                k = k.replace("lycoris_", "")
                k = f"lora_{key}_{k}"
                new_state_dict[k] = v

            state_dict.update(new_state_dict)

        if cfg.get("save_format") == "safetensors":
            model_path += ".safetensors"
            save_file(state_dict, model_path, metadata=metadata)
        else:
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        logger.info(f"Saved model to {model_path}")
