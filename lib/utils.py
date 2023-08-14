import os
import re
import torch

from io import BytesIO

from omegaconf import OmegaConf
import requests
import torch
from transformers import (
    CLIPTextModel, 
    CLIPTextConfig,
    CLIPTokenizer
)
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config
)
from . import diffusers_convert
from lightning.pytorch.utilities import rank_zero_only

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)

def get_world_size() -> int:
    # if config is not None and config.lightning.accelerator == "tpu":
    #     import torch_xla.core.xla_model as xm
    #     return xm.xrt_world_size()
    
    return int(os.environ.get("WORLD_SIZE", 1))

def get_local_rank() -> int:
    # if config.lightning.accelerator == "tpu":
    #     import torch_xla.core.xla_model as xm
    #     return xm.get_ordinal()
    
    return int(os.environ.get("LOCAL_RANK", -1))

def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict

def state_dict_prefix_replace(state_dict, replace_prefix):
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            state_dict[x[1]] = state_dict.pop(x[0])
    return state_dict

def convert_to_sd(state_dict):
    unet_state_dict = {k.replace("unet.", "", 1): v for k, v in state_dict.items() if k.startswith("unet")}
    vae_state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items() if k.startswith("vae")}
    text_enc_dict = {k.replace("text_encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("text_encoder")}
    text_enc_dict_2 = {k.replace("text_encoder_2.", "", 1): v for k, v in state_dict.items() if k.startswith("text_encoder_2")}
        
    # Convert the UNet model
    unet_state_dict = diffusers_convert.convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

    # Convert the VAE model
    vae_state_dict = diffusers_convert.convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
    is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

    if is_v20_model:
        # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
        text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()}
        text_enc_dict = diffusers_convert.convert_text_enc_state_dict_v20(text_enc_dict)
        text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
    else:
        text_enc_dict = diffusers_convert.convert_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}
        
    if len(text_enc_dict_2) > 0:
        pass

    # Put together new checkpoint
    return {**unet_state_dict, **vae_state_dict, **text_enc_dict}

def convert_to_df(checkpoint, return_pipe=False):
    key_name_v2_1 = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
    key_name_sd_xl_base = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
    key_name_sd_xl_refiner = "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias"
    
    global_step = None
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]

    # model_type = "v1"
    config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    upcast_attention = None
    if key_name_v2_1 in checkpoint and checkpoint[key_name_v2_1].shape[-1] == 1024:
        # model_type = "v2"
        config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"

        if global_step == 110000:
            # v2.1 needs to upcast attention
            upcast_attention = True
    elif key_name_sd_xl_base in checkpoint:
        # only base xl has two text embedders
        config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    elif key_name_sd_xl_refiner in checkpoint:
        # only refiner xl has embedder and one text embedders
        config_url = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"

    original_config_file = BytesIO(requests.get(config_url).content)
    original_config = OmegaConf.load(original_config_file)
    
    # Convert the text model.
    if (
        "cond_stage_config" in original_config.model.params
        and original_config.model.params.cond_stage_config is not None
    ):
        model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    elif original_config.model.params.network_config is not None:
        if original_config.model.params.network_config.params.context_dim == 2048:
            model_type = "SDXL"
        else:
            model_type = "SDXL-Refiner"
        image_size = 1024 

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
            # as it relies on a brittle global step parameter here
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
        if image_size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            image_size = 512 if global_step == 875000 else 768
    else:
        prediction_type = "epsilon"
        image_size = 512 

    num_train_timesteps = getattr(original_config.model.params, "timesteps", None) or 1000
    if model_type in ["SDXL", "SDXL-Refiner"]:
        scheduler_dict = {
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "interpolation_type": "linear",
            "num_train_timesteps": num_train_timesteps,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "timestep_spacing": "leading",
        }
        scheduler = DDIMScheduler.from_config(scheduler_dict)
        # scheduler_type = "euler"
    else:
        beta_start = getattr(original_config.model.params, "linear_start", None) or 0.02
        beta_end = getattr(original_config.model.params, "linear_end", None) or 0.085
        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
    # make sure scheduler works correctly with DDIM
    scheduler.register_to_config(clip_sample=False)

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config, image_size=image_size)
    unet_config["upcast_attention"] = upcast_attention
    unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config, extract_ema=False)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    if model_type == "FrozenOpenCLIPEmbedder":
        text_model = convert_open_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    elif model_type == "FrozenCLIPEmbedder":
        keys = list(checkpoint.keys())
        text_model_dict = {}
        for key in keys:
            if key.startswith("cond_stage_model.transformer"):
                dest_key = key[len("cond_stage_model.transformer.") :]
                if "text_model" not in dest_key:
                    dest_key = f"text_model.{dest_key}"
                text_model_dict[dest_key] = checkpoint[key]
        
        text_model = CLIPTextModel(CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14"))
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if "text_model.embeddings.position_ids" not in text_model.state_dict().keys() \
            and "text_model.embeddings.position_ids" in text_model_dict.keys():
            del text_model_dict["text_model.embeddings.position_ids"] 

        if len(text_model_dict) < 10:
            text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    elif model_type in ["SDXL", "SDXL-Refiner"]:
        tokenizer = None
        text_encoder = None
        tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", pad_token="!")
        text_encoder_2 = convert_open_clip_checkpoint(
            checkpoint, 
            config_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            prefix="conditioner.embedders.1.model." if model_type == "SDXL" else "conditioner.embedders.0.model.",
            has_projection=True, 
            projection_dim=1280,
        )
        if model_type == "SDXL":
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = convert_ldm_clip_checkpoint(checkpoint)
    
    if not return_pipe:
        if model_type in ["SDXL", "SDXL-Refiner"]:
            return converted_unet_checkpoint, converted_vae_checkpoint, text_encoder, text_encoder_2
        else:
            return converted_unet_checkpoint, converted_vae_checkpoint, text_model_dict
    else:
        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
        unet.load_state_dict(converted_unet_checkpoint)
        
        if model_type in ["SDXL", "SDXL-Refiner"]:
            return StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=True,
                add_watermarker=False,
            )
        else:
            text_model.load_state_dict(text_model_dict)
            return StableDiffusionPipeline(
                unet=unet,
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
