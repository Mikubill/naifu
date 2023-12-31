import functools
import hashlib
import json
import math
import cv2
import h5py as h5
import numpy as np
import random
import torch

from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Generator, Optional# type: ignore
from data.processors import placebo
from data.embeddings import get_size_embeddings
from torchvision import transforms
from lib.logging import logger


image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def is_img(path: Path):
    return path.suffix in image_suffix


def sha1sum(txt):
    return hashlib.sha1(txt.encode()).hexdigest()


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    original_size: tuple[int, int] # h, w
    cropped_size: Optional[tuple[int, int]] # h, w
    # mask: torch.Tensor | None = None


def dirwalk(
    path: Path, cond: Optional[Callable] = None
) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class StoreBase(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        root_path,
        ucg=0,
        rank=0,
        prompt_processor: str | None = None,
        dtype=torch.float16,
        use_central_crop=False,
        **kwargs,
    ):
        self.ucg = ucg
        self.rank = rank
        self.root_path = Path(root_path)
        self.dtype = dtype
        self.use_central_crop = use_central_crop
        self.kwargs = kwargs

        self.length = 0
        self.rand_list: list = []
        self.to_ratio: None | np.ndarray = None
        self.raw_res: list[tuple[int, int]] = []
        self.curr_res: list[tuple[int, int]] = []
        
        assert self.root_path.exists()
        if prompt_processor:
            self.prompt_processor = get_class(prompt_processor)
        else:
            self.prompt_processor = placebo
            
        self.embeds_cache = kwargs.get("precompute_embeds", None)
        self.embeds_cache_keys = None
        if self.embeds_cache:
            self.embeds_cache = h5.File(self.embeds_cache, "r", libver="latest")
            self.embeds_cache_keys = {k[:-5] for k in self.embeds_cache.keys() if k.endswith(".emb1")}

    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, str, (int, int)]:
        raise NotImplementedError

    def fix_aspect_randomness(self, rng: np.random.Generator):
        raise NotImplementedError
    
    @staticmethod
    @functools.cache
    def fit_dimensions(target_ratio, min_h, min_w):
        min_area = min_h * min_w
        h = max(min_h, math.ceil(math.sqrt(min_area * target_ratio)))
        w = max(min_w, math.ceil(h / target_ratio))
        
        if w < min_w:
            w = min_w
            h = max(min_h, math.ceil(w * target_ratio))
        
        while h * w < min_area:
            increment = 8
            if target_ratio >= 1:
                h += increment
            else:
                w += increment

            w = max(min_w, math.ceil(h / target_ratio))
            h = max(min_h, math.ceil(w * target_ratio))
        return int(h), int(w)

    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_ratio is not None, "to_ratio is not initialized"
        H, W = entry.pixel.shape[-2:]
        base_ratio = H / W
        target_ratio = self.to_ratio[i]
        h, w = self.ratio_to_bucket[target_ratio]
        if not entry.is_latent:
            resize_h, resize_w = self.fit_dimensions(base_ratio, H, W)
            interp = cv2.INTER_AREA if resize_h < H else cv2.INTER_CUBIC
            entry.pixel = cv2.resize(entry.pixel, (resize_w, resize_h, ), interpolation=interp)
        else:
            h, w = h // 8 , w // 8 
        
        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - h) // 2, (W - w) // 2
        else:
            assert H >= h and W >= w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H - h), random.randint(0, W - w)
            
        entry.pixel = entry.pixel[:, dh : dh + h, dw : dw + w]
        return entry, dh, dw

    @torch.no_grad()
    def get_batch(self, indices: list[int]) -> Entry:
        entries = [self._get_entry(i) for i in indices]
        crop_pos = []
        pixels = []
        prompts = []
        original_sizes = []
        cropped_sizes = []
        hidden1, hidden2, pool2 = [], [], []

        for e, i in zip(entries, indices):
            e, dh, dw = self.crop(e, i)
            pixels.append(e.pixel)
            original_size = torch.asarray(e.original_size)
            original_sizes.append(original_size)

            cropped_size = e.pixel.shape[-2:]
            cropped_size = (cropped_size[0] * 8, cropped_size[1] * 8) if e.is_latent else cropped_size
            cropped_size = torch.asarray(cropped_size)
            cropped_sizes.append(cropped_size)
            
            cropped_pos = (dh, dw)
            cropped_pos = (cropped_pos[0] * 8, cropped_pos[1] * 8) if e.is_latent else cropped_pos
            cropped_pos = torch.asarray(cropped_pos)
            crop_pos.append(torch.asarray(cropped_pos))
            
            hashkey = sha1sum(e.prompt)
            if self.embeds_cache is not None:
                assert hashkey in self.embeds_cache_keys
                hashkey = sha1sum(e.prompt)
                hidden1.append(torch.asarray(self.embeds_cache[f"{hashkey}.emb1"][:]))
                hidden2.append(torch.asarray(self.embeds_cache[f"{hashkey}.emb2"][:]))
                pool2.append(torch.asarray(self.embeds_cache[f"{hashkey}.pool2"][:]))
            else:
                prompts.append(e.prompt)
            
        is_latent = entries[0].is_latent
        shape = entries[0].pixel.shape

        for e in entries[1:]:
            assert e.is_latent == is_latent
            assert e.pixel.shape == shape, f"{e.pixel.shape} != {shape} for the same batch"

        pixel = torch.stack(pixels, dim=0).contiguous()
        cropped_sizes = torch.stack(cropped_sizes)
        original_sizes = torch.stack(original_sizes)
        crop_pos = torch.stack(crop_pos)
        
        if self.embeds_cache is not None:
            hidden1, hidden2, pool2 = torch.stack(hidden1), torch.stack(hidden2), torch.stack(pool2)
            if hidden1.shape[1] == 1:
                hidden1, hidden2, pool2 = hidden1.squeeze(1), hidden2.squeeze(1), pool2.squeeze(1)
            size_embs = get_size_embeddings(original_sizes, crop_pos, cropped_sizes, hidden1.device)
            vector_embedding = torch.cat([pool2, size_embs], dim=1)
            text_embedding = torch.cat([hidden1, hidden2], dim=2)
            prompts = {"crossattn": text_embedding, "vector": vector_embedding}
            
        return {
            "prompts": prompts,
            "pixels": pixel,
            "is_latent": is_latent,
            "is_cached_embeds": self.embeds_cache_keys is not None,
            "target_size_as_tuple": cropped_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_pos,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_entry(self, index) -> Entry:
        is_latent, pixel, prompt, original_size = self.get_raw_entry(index)

        if self.ucg and np.random.rand() < self.ucg:
            prompt = ""
            
        prompt, _ = self.prompt_processor(prompt)    
        pixel = pixel.to(dtype=self.dtype)
        shape = pixel.shape
        if shape[-1] == 3 and shape[-1] < shape[0] and shape[-1] < shape[1]:  
            pixel = pixel.permute(2, 0, 1) # HWC -> CHW
            
        return Entry(is_latent, pixel, prompt, original_size, None)
    
    def repeat_entries(self, k, res, index=None):
        repeat_strategy = self.kwargs.get("repeat_strategy", None)
        if repeat_strategy is not None:
            assert index is not None
            index_new = index.copy()
            for i, ent in enumerate(index):
                for strategy, mult in repeat_strategy:
                    if strategy in str(ent):
                        k = k + [k[i]] * (mult-1)
                        res = res + [res[i]] * (mult-1)
                        index_new = index_new + [index_new[i]] * (mult-1)
                        break
        else:
            index_new = index
        return k, res, index_new


class LatentStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt_mapping = next(dirwalk(self.root_path, lambda p: p.suffix == ".json"))
        prompt_mapping = json.loads(Path(prompt_mapping).read_text())
        
        self.h5_paths = list(dirwalk(self.root_path, lambda p: p.suffix == ".h5" and "prompt_cache" not in p.stem))
        self.h5_keymap = {}
        self.h5_filehandles = {}
        self.paths = []
        for h5_path in tqdm(
            self.h5_paths,
            desc="Loading latents",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        ):
            fs = h5.File(h5_path, "r", libver="latest")
            for k in fs.keys():
                hashkey = k[:-8] # ".latents"
                assert hashkey in prompt_mapping, f"Key {k} not found in prompt_mapping"
                it = prompt_mapping[hashkey]
                if not it["train_use"]: continue
                prompt, it_path = it["train_caption"], it["file_path"]
                height, width = it["train_height"], it["train_width"]
                self.paths.append(it_path)
                self.raw_res.append((height, width))
                self.h5_keymap[k] = (h5_path, prompt, (height, width))
        
        self.keys = list(self.h5_keymap.keys())
        self.length = len(self.keys)
        logger.debug(f"Loaded {self.length} latent codes from {self.root_path}")
        
        self.keys, self.raw_res, self.paths = self.repeat_entries(self.keys, self.raw_res, index=self.paths)
        self.length = len(self.paths)
        logger.debug(f"Using {self.length} entries after applied repeat strategy")
        
    def setup_filehandles(self):
        self.h5_filehandles = {}
        for h5_path in self.h5_paths:
            self.h5_filehandles[h5_path] = h5.File(h5_path, "r", libver="latest")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        if len(self.h5_filehandles) == 0:
            self.setup_filehandles()
        latent_key = self.keys[index]
        h5_path, prompt, original_size = self.h5_keymap[latent_key]
        latent = torch.asarray(self.h5_filehandles[h5_path][latent_key][:]).float()
        scaled = self.h5_filehandles[h5_path][latent_key].attrs.get("scale", True)
        latent = latent * 0.13025 if not scaled else latent
        return True, latent, prompt, original_size


class DirectoryImageStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_ext = self.kwargs.get("label_ext", ".txt")
        self.paths = list(dirwalk(self.root_path, is_img))
        self.length = len(self.paths)
        logger.debug(f"Found {self.length} images in {self.root_path}")

        cloned_paths = self.paths.copy()
        for p in tqdm(
            self.paths, desc="Loading image sizes", leave=False, ascii=True,
        ):
            try:
                w, h = Image.open(p).size
                self.raw_res.append((h, w))
            except Exception as e:
                logger.warn(f"Error processing {p}: {e}")
                cloned_paths.remove(p)
                
        self.paths = cloned_paths
        self.length = len(self.paths)
        self.prompts: list[str] = []
        for path in tqdm(
            self.paths,
            desc="Loading prompts",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        ):
            p = path.with_suffix(label_ext)
            with open(p, "r") as f:
                self.prompts.append(f.read())
        logger.debug(f"Loaded {len(self.prompts)} prompts")
            
        self.base_len = self.kwargs["base_len"]
        logger.debug(f"Loaded {self.length} image sizes")
        
        self.prompts, self.raw_res, self.paths = self.repeat_entries(self.prompts, self.raw_res, index=self.paths)
        self.length = len(self.paths)
        logger.debug(f"Using {self.length} entries after applied repeat strategy")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        p = self.paths[index]
        prompt = self.prompts[index]
        _img = Image.open(p)
        if _img.mode == "RGB":
            img = np.array(_img)
        elif _img.mode == "RGBA":
            img = np.array(_img)
            rgb, alpha = img[:, :, :3], img[:, :, 3:]
            fp_alpha = alpha / 255
            rgb[:] = rgb * fp_alpha + (255 - alpha)
            img = rgb
        else:
            img = np.array(_img.convert("RGB"))
            
        img = IMAGE_TRANSFORMS(img)
        h, w = img.shape[-2:]
        return False, img, prompt, (h, w)