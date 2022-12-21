
import argparse
import os
import pickle
from pathlib import Path

import torch
from data.buckets import AspectRatioBucket
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


def transformer(img, size, center_crop=False):
    x, y = img.size
    short, long = (x, y) if x <= y else (y, x)

    w, h = size
    min_crop, max_crop = (w, h) if w <= h else (h, w)
    ratio_src, ratio_dst = float(long / short), float(max_crop / min_crop)

    if ratio_src > ratio_dst:
        new_w, new_h = (min_crop, int(min_crop * ratio_src)) if x < y else (int(min_crop * ratio_src), min_crop)
    elif ratio_src < ratio_dst:
        new_w, new_h = (max_crop, int(max_crop / ratio_src)) if x > y else (int(max_crop / ratio_src), max_crop)
    else:
        new_w, new_h = w, h

    image_transforms = transforms.Compose([
        transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop((h, w)) if center_crop else transforms.RandomCrop((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return image_transforms(img)

def read_img(x):
    img = Image.open(x)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    return img

def resolver(x):
    fp = os.path.splitext(x)[0]
    with open(fp + ".txt") as f:
        prompt = f.read()
    return read_img(x), prompt

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("-d", "--data", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="latent_dataset.pth")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    cuda = torch.device("cuda")
    assert args.data != None
    
    vae = AutoencoderKL.from_pretrained(config.trainer.model_path, subfolder="vae") 
    vae.to(cuda)
    vae.eval()  
    id_size_map = {}
    id_prompt_map = {}
    latent_dataset = {}

    for x in tqdm(Path(args.data).iterdir(), desc=f"Loading resolutions"):
        if not (x.is_file() and x.suffix in [".jpg", ".png", ".webp", ".bmp", ".gif", ".jpeg", ".tiff"]):
            continue
        img, prompt = resolver(x)
        size = img.size
        id_size_map[x] = size
        id_prompt_map[x] = prompt
    
    dataset = AspectRatioBucket(id_size_map, **config.arb)
    for entry in dataset.buckets.keys():
        size = dataset.resolutions[entry]
        imgs = dataset.buckets[entry][:]
        for img in tqdm(imgs, total=len(imgs), desc=f"Working on {size}"):
            img_data = read_img(img)
            img_data_actual = torch.stack([transformer(img_data, size, config.dataset.center_crop)]).to(cuda).float()
            img_data_base = torch.stack([transformer(img_data, config.arb.base_res, True)]).to(cuda).float()
            latents = vae.encode(img_data_actual).latent_dist.sample() * 0.18215
            latents_base = vae.encode(img_data_base).latent_dist.sample() * 0.18215
            latent_dataset[img] = {
                "latents_actual": latents.detach(),
                "latents_base": latents_base.detach(),
                "prompts": id_prompt_map[img],
                "size": size,
            }
            del img_data, img_data_actual, img_data_base
            
    torch.save(latent_dataset, args.output, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            