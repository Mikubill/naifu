# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """

import argparse
import os

import torch


try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import BertTokenizerFast, CLIPTokenizer
from lib.utils import (
    convert_ldm_openclip_checkpoint,
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
    create_ldm_bert_config,
    convert_ldm_bert_checkpoint,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        "--src",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--vae_path", default=None, type=str, help="Path to the vae to convert."
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancest', 'dpm']",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--dump_path",
        "--dst",
        default=None,
        type=str,
        required=True,
        help="Path to the output model.",
    )

    args = parser.parse_args()

    if args.original_config_file is None:
        os.system(
            "wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
        )
        args.original_config_file = "./v1-inference.yaml"

    original_config = OmegaConf.load(args.original_config_file)

    checkpoint = torch.load(args.checkpoint_path)
    checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end
    if args.scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif args.scheduler_type == "lms":
        from diffusers import LMSDiscreteScheduler

        scheduler = LMSDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "euler":
        from diffusers import EulerDiscreteScheduler

        scheduler = EulerDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "euler-ancestral":
        from diffusers import EulerAncestralDiscreteScheduler

        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "dpm":
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif args.scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        raise ValueError(f"Scheduler of type {args.scheduler_type} doesn't exist!")

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        checkpoint, unet_config, path=args.checkpoint_path, extract_ema=args.extract_ema
    )

    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model.
    if args.vae_path:
        vae_config = create_vae_diffusers_config(original_config)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config, vae=args.vae_path)
    else:
        vae_config = create_vae_diffusers_config(original_config)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model.
    text_model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    if text_model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        # feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    elif text_model_type == "FrozenOpenCLIPEmbedder":
        text_model = convert_ldm_openclip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    else:
        text_config = create_ldm_bert_config(original_config)
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        pipe = LDMTextToImagePipeline(
            vqvae=vae,
            bert=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    pipe.save_pretrained(args.dump_path)
