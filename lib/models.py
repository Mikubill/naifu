import functools
from pathlib import Path
import tarfile

import requests
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.optimization import (
    get_scheduler,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


def download_model(url, model_path="model"):
    print(f'Downloading: "{url}" to {model_path}\n')
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get("content-length", 0))
    
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  
    with tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:    
        file = tarfile.open(fileobj=r_raw, mode="r|gz")
        file.extractall(path=model_path) 

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def load_models(model_path, config):
    model_url = config.trainer.model_url

    if not Path(model_path).is_dir() or not (Path(model_path) / "model_index.json").is_file():
        Path(model_path).mkdir(exist_ok=True)
        download_model(model_url, model_path)

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    noise_scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler")

    optimizer = get_class(config.optimizer.name)(unet.parameters(), **config.optimizer.params)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, **config.scheduler.params)
    
    return (
        tokenizer,
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        optimizer,
        scheduler,
    )
