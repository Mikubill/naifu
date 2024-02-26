import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from common.utils import load_torch_file, get_class, EmptyInitWrapper, rank_zero_print, rank_zero_only
from common.dataset import AspectRatioDataset, worker_init_fn
from modules.train_sdxl import SupervisedFineTune
from modules.sdxl_utils import get_hidden_states_sdxl, convert_sdxl_text_encoder_2_checkpoint
from modules.sdxl_utils import get_size_embeddings
from modules.scheduler_utils import apply_zero_terminal_snr, cache_snr_values
from lightning.pytorch.utilities.model_summary import ModelSummary
from safetensors.torch import save_file

from diffusers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

try:
    from lycoris import create_lycoris, LycorisNetwork
except ImportError as e:
    raise ImportError(f"\n\nError import lycoris: {e} \nTry install lycoris using `pip install lycoris_lora toml`")

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path
    model = StableDiffusionModel(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )
    dataset = AspectRatioDataset(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        base_len=config.trainer.resolution,
        **config.dataset,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=None,
        persistent_workers=False,
        num_workers=config.dataset.get("num_workers", 4),
        worker_init_fn=worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )
    
    params_to_optim = [{"params": model.lycoris_unet.parameters()}]
    # params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.lycoris_te1.parameters(), "lr": lr})
            
    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.lycoris_te2.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(
        params_to_optim, **optim_param
    )
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
        
    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")
        
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    return model, dataset, dataloader, optimizer, scheduler


def init_text_encoder():
    text_model_1_cfg = CLIPTextConfig.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder")
    text_model_2_cfg = CLIPTextConfig.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2")
    
    with EmptyInitWrapper():    
        text_model_1 = CLIPTextModel._from_config(text_model_1_cfg)
        text_model_2 = CLIPTextModelWithProjection._from_config(text_model_2_cfg)
        
    tokenizer1 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    text_model_1 = text_model_1.requires_grad_(False)
    text_model_2 = text_model_2.requires_grad_(False)
    return text_model_1, text_model_2, tokenizer1, tokenizer2


# define the LightningModule
class StableDiffusionModel(SupervisedFineTune):
    def init_model(self):
        advanced = self.config.get("advanced", {})
        sd = load_torch_file(self.model_path, self.target_device)
        vae, unet, _ = self.build_models(init_conditioner=False)
        self.first_stage_model = vae
        self.model = unet
        
        te1_sd, te2_sd, unet_sd, vae_sd = {}, {}, {}, {}
        for k in list(sd.keys()):
            if k.startswith("conditioner.embedders.0.transformer."):
                te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = sd.pop(k)
            elif k.startswith("conditioner.embedders.1.model."):
                te2_sd[k] = sd.pop(k)
            elif k.startswith("first_stage_model."):
                vae_sd[k.replace("first_stage_model.", "")] = sd.pop(k)
            elif k.startswith("model.diffusion_model."):
                unet_sd[k.replace("model.diffusion_model.", "diffusion_model.")] = sd.pop(k)
                
        converted_sd = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
        self.text_encoder_1, self.text_encoder_2, self.tokenizer_1, self.tokenizer_2 = init_text_encoder()
        self.text_encoder_1.load_state_dict(te1_sd, strict=False)
        self.text_encoder_2.load_state_dict(converted_sd)
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
        self.init_lycoris()
        
    def init_lycoris(self):
        cfg = self.config 
        default_cfg = cfg.lycoris    
        self.lycoris_mapping = lycoris_mapping = {}
        self.model.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        LycorisNetwork.apply_preset({"target_name": ".*"})
        
        self.lycoris_unet = create_lycoris(self.model.diffusion_model, **cfg.get("lycoris_unet", default_cfg))
        self.lycoris_te1 = create_lycoris(self.text_encoder_1, **cfg.get("lycoris_te1", default_cfg))
        self.lycoris_te2 = create_lycoris(self.text_encoder_2, **cfg.get("lycoris_te2", default_cfg))
        lycoris_mapping["unet"] = self.lycoris_unet
        lycoris_mapping["te1"] = self.lycoris_te1
        lycoris_mapping["te2"] = self.lycoris_te2
        
        self.lycoris_unet.to(self.target_device).apply_to()
        self.lycoris_unet.requires_grad_(True)
        
        if not self.config.advanced.get("train_text_encoder_1"):
            self.lycoris_te1.to(self.target_device).apply_to()
            self.lycoris_te1.requires_grad_(True)
            
        if not self.config.advanced.get("train_text_encoder_2"):
            self.lycoris_te2.to(self.target_device).apply_to()
            self.lycoris_te2.requires_grad_(True)
        
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
            self.fabric.device
        )
        cond = {
            "crossattn": torch.cat([hidden1, hidden2], dim=2),
            "vector": torch.cat([pooled, emb], dim=1),
        }
        return cond
       
    @rank_zero_only 
    def save_checkpoint(self, model_path):
        cfg = self.config.trainer
        string_cfg = OmegaConf.to_yaml(self.config)
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
            save_file(state_dict, model_path, metadata={"trainer_config": string_cfg})
        else:
            model_path += ".ckpt"
            torch.save(state_dict, model_path)
        rank_zero_print(f"Saved model to {model_path}")