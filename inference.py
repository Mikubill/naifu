# python trainer.py --model_path=/tmp/model --config config/test.yaml
import argparse
import copy
import os

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
from omegaconf import OmegaConf
from pathlib import Path
from lib.model import StableDiffusionModel
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a girl running fast on the country lane",
    )
    args = parser.parse_args()
    return args


def setup_torch(config):
    major, minor = torch.__version__.split('.')[:2]
    if (int(major) > 1 or (int(major) == 1 and int(minor) >= 12)) and torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{device.major}.{device.minor}")
        precision = 'high' if config.lightning.precision == 32 else 'medium'
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)


def sampler(model, save_dir, prompts, negative_prompts, ):
    if not any(prompts):
        return
        
    save_dir = Path(save_dir) 
    save_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(int(time.time()))
    
    images = []
    images.extend(model.sample(prompts, negative_prompts, generator))

    for j, image in enumerate(images):
        image.save(save_dir / f"nd_sample_{j}.png")
        
def main(args):
    model = StableDiffusionModel(args.path, OmegaConf.create({"trainer": {"use_xformers": False, "use_ema": False}}))    
    model.model.eval().cuda()
    model.requires_grad_(False)
    
    negative_prompts = "lowres, low quality, text, error, extra digit, cropped"
    sampler(model, ".", args.prompt, negative_prompts)

if __name__ == "__main__":
    args = parse_args()
    main(args)
