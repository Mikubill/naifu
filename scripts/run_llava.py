# cp run_llava.py .. && cd ..
# python run_llava.py

import re
from PIL import Image
from io import BytesIO
import requests
import torch
from transformers import AutoTokenizer

from data.conversation import conv_templates
from models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from models.llava.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from models.llava.mm_utils import process_images, tokenizer_image_token

from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List, Optional


from omegaconf import OmegaConf

config_dict = {
    "version": "chatml_direct",
    "image_aspect_ratio": "anyres",
    "mm_hidden_size": 1024,
    "mm_patch_merge_type": "spatial_unpad",
    "mm_projector_lr": 2e-5,
    "mm_projector_type": "mlp2x_gelu",
    "mm_resampler_type": None,
    "mm_use_im_patch_token": False,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "patch",
    "mm_vision_select_layer": -2,
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "pretrain_mm_mlp_adapter": None,
    "mm_vision_tower_lr": 2e-6,
    "tune_mm_mlp_adapter": True,
    "tune_mm_vision_tower": True,
    "freeze_backbone": False,
    "image_grid_pinpoints": [
        [336, 672],
        [672, 336],
        [672, 672],
        [1008, 336],
        [336, 1008],
    ],
}

mm_config = OmegaConf.create(config_dict)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(model, tokenizer, args):
    model_name = args.model_name
    image_processor = model.get_vision_tower().image_processor
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.bfloat16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)
    
    
if __name__ == "__main__":
    model_path = "liuhaotian/llava-v1.6-34b"
    prompt = "What are the things I should be cautious about when I visit here? in chinese"
    image_file = "https://llava-vl.github.io/static/images/view.jpg"
    
    cfg_pretrained = LlavaConfig.from_pretrained(model_path)
    cfg_pretrained.update(mm_config)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=cfg_pretrained,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    print(f"Query: {prompt}")
    args = OmegaConf.create({
        "model_name": model_path,
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 1024,
    })
    eval_model(model, tokenizer, args)
