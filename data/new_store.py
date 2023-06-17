import itertools
import json
import os
import random
import torch
import tempfile
import os, binascii

from lib.augment import AugmentTransforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from tqdm import tqdm
from lib.utils import get_local_rank
import numpy as np
import re
import h5py as h5
from dataclasses import dataclass

from typing import Callable, Optional, Generator
from transformers import CLIPTokenizer
import data.custom_prompt_processors as custom_prompt_processors
import cv2
from typing import Any


image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])
m_repeat = re.compile(r'^r=(\d+)')
def is_img(path: Path):
    return path.suffix in image_suffix


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor
    input_ids: torch.Tensor
    mask: torch.Tensor|None = None


    
def dirwalk(path: Path, cond: Optional[Callable] = None, mult:int=1) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            matched = m_repeat.match(p.name)
            if matched:
                x = int(matched.group(1))
                assert x>=0
            else:
                x = 1
            yield from dirwalk(p, cond, mult*x)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            for _ in range(mult):
                yield p


# From https://stackoverflow.com/a/16778797/10444046
def rotatedRectWithMaxArea(h, w, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(hr), int(wr)

class StoreBase(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        root_path,
        tokenizer:CLIPTokenizer,
        max_length=225,
        ucg=0,
        rank=0,
        prompt_processor:str|None=None,
        enable_mask=False,
        dtype=torch.float16,
        use_central_crop=False,
        **kwargs
    ):
        self.ucg = ucg
        self.max_length = max_length
        self.tokenizer=tokenizer
        self.rank = rank
        self.root_path = Path(root_path)
        self.enable_mask = enable_mask
        self.dtype = dtype
        self.use_central_crop = use_central_crop
        self.kwargs = kwargs
    
        self.img_length = 0
        self.img_length = None

        self.rand_list: list = []
        self.to_ratio: None|np.ndarray = None
        self.raw_res: list[tuple[int, int]] = []
        self.curr_res:list[tuple[int, int]] = []
        assert self.root_path.exists()
        if prompt_processor:
            assert hasattr(custom_prompt_processors, prompt_processor), f"Specified prompt processor {prompt_processor} is not found in custom_prompt_processors.py"
            self.prompt_processor = getattr(custom_prompt_processors, prompt_processor)
        else:
            self.prompt_processor = None
        self.init()
        
    def init(self):
        raise NotImplementedError
    
    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, Optional[np.ndarray], str]:
        raise NotImplementedError
    
    def fix_aspect_randomness(self, rng:np.random.Generator):
        raise NotImplementedError

    def crop(self, entry: Entry, i:int) -> Entry:
        
        assert self.to_ratio is not None, "to_ratio is not initialized"
        ratio = self.to_ratio[i]
        H, W = entry.pixel.shape[1:]
        rnd_factor = 8 if entry.is_latent else 64
        h = H if H<W else int(W*ratio/rnd_factor)*rnd_factor
        w = W if W<H else int(H/ratio/rnd_factor)*rnd_factor
        if self.use_central_crop:
            dh, dw = (H-h)//2, (W-w)//2
        else:
            assert H>=h and W>=w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H-h), random.randint(0, W-w)
        entry.pixel = entry.pixel[:, dh:dh+h, dw:dw+w]
        if entry.mask is not None:
            entry.mask = entry.mask[dh:dh+h, dw:dw+w]
        return entry
    
    def get_batch(self, indices:list[int]) -> Entry:
        entries = [self._get_entry(i) for i in indices]
        for e, i in zip(entries, indices):
            self.crop(e, i)
        is_latent = entries[0].is_latent
        shape = entries[0].pixel.shape
        any_has_mask = any(e.mask is not None for e in entries)
        for e in entries[1:]:
            assert e.is_latent == is_latent
            assert e.pixel.shape == shape, f"{e.pixel.shape} != {shape} for the same batch"
        if any_has_mask:
            mask = torch.ones((len(entries), *shape[1:]), dtype=self.dtype)
            for i, e in enumerate(entries):
                if e.mask is not None:
                    mask[i] = e.mask
        else:
            mask = None
        pixel = torch.stack([e.pixel for e in entries], dim=0)
        # make pixel C contiguous
        pixel = pixel.contiguous()
        input_ids = [e.input_ids for e in entries] # type: ignore
        input_ids: torch.Tensor = self.tokenizer.pad(
            encoded_inputs = {"input_ids": input_ids}, # type: ignore
            padding=True, return_tensors="pt"
        ).input_ids
        return Entry(is_latent, pixel, input_ids, mask)
    
    def tokenize(self, prompt:str) -> torch.Tensor:
        '''
        Handle token truncation in collate_fn()
        '''
        return self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
        ).input_ids
        
    def __len__(self):
        return self.img_length

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_entry(self, index) -> Entry:
        is_latent, pixel, mask, prompt = self.get_raw_entry(index)
        
        if self.ucg and np.random.rand() < self.ucg:
            prompt=''
        elif self.prompt_processor is not None:
            prompt = self.prompt_processor(prompt)
        input_ids = self.tokenize(prompt)
        
        pixel = torch.from_numpy(pixel).to(dtype=self.dtype)
        shape = pixel.shape
        if shape[-1]==3 and shape[-1]<shape[0] and shape[-1]<shape[1]: # HWC -> CHW
            pixel = pixel.permute(2, 0, 1)
        mask = torch.from_numpy(mask).to(dtype=self.dtype) if mask is not None else None
        return Entry(is_latent, pixel, input_ids, mask)
    


class LatentStore(StoreBase):
    
    def init(self):
        assert self.root_path.suffix == ".h5"
        self.h5f = h5.File(self.root_path, "r")
        self.keys = list(self.h5f.keys())
        self.img_length = len(self.keys)

        self.probs = [np.array(self.h5f[k]["probs"][:]) for k in self.keys] # type: ignore
        print(f"Loaded {self.img_length} latent codes from {self.root_path}")
    
    def fix_aspect_randomness(self, rng:np.random.Generator):
        self.rand_list:list[int] = [rng.choice(len(prob), p=prob) for prob in self.probs]
        self.curr_res = [
            self.h5f[k][f'{i}.pixel'].shape[1:] for k, i in zip(self.keys, self.rand_list) # type: ignore
        ]
    
    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, np.ndarray | None, str]:
        aug_idx = self.rand_list[index]
        g:h5.Group = self.h5f[self.keys[index]] # type: ignore
        
        pixel:np.ndarray = np.array(g[f'{aug_idx}.pixel']) # type: ignore
        mask = None
        if self.enable_mask and f'{aug_idx}.mask' in g:
            mask = np.array(g[f'{aug_idx}.mask']) # type: ignore
        prompt:str = g.attrs['prompt'] # type: ignore
        return True, pixel, mask, prompt



class DirectoryImageStore(StoreBase):
    
    def init(self):
        label_ext = self.kwargs.get("label_ext", ".txt")
        self.paths = list(dirwalk(self.root_path, is_img))
        self.img_length = len(self.paths)
        print(f"Found {self.img_length} images in {self.root_path}")
        
        self.prompts:list[str] = []
        for path in tqdm(self.paths, desc="Loading prompts", disable=self.rank!=0, leave=False, ascii=True):
            p = path.with_suffix(label_ext)
            with open(p, "r") as f:
                self.prompts.append(f.read())
        print(f"Loaded {len(self.prompts)} prompts")
        for p in tqdm(self.paths, desc="Loading image sizes", disable=self.rank!=0, leave=False, ascii=True):
            w,h = Image.open(p).size
            self.raw_res.append((h, w))
        self.base_len = self.kwargs["base_len"]
        
        print(f"Loaded {len(self.raw_res)} image sizes")


    def augment(self, image:np.ndarray, mask:None|np.ndarray, index:int) -> tuple[np.ndarray, np.ndarray|None]:
        if not self.kwargs.get('augmentation'):
            return image, mask
        if self.kwargs['augmentation'].get('rotate'):
            angle = self.rand_list[index]
            h,w = self.curr_res[index]
            H,W = self.raw_res[index]
            rot = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
            dw, dh = (W-w)//2, (H-h)//2
            image = cv2.warpAffine(image, rot, (W,H))[dh:dh+h, dw:dw+w]
            if mask is not None:
                mask = cv2.warpAffine(mask, rot, (W,H))[dh:dh+h, dw:dw+w]
        if p:=self.kwargs['augmentation'].get('flip'):
            if random.random() < p:
                image = cv2.flip(image, 1)
                if mask is not None:
                    mask = cv2.flip(mask, 1)
        return image, mask
            

    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, np.ndarray | None, str]:
        p = self.paths[index]
        prompt = self.prompts[index]
        _img = Image.open(p)
        mask = None
        if _img.mode == "RGB":
            img = np.array(_img)
        elif _img.mode == "RGBA":
            if not self.enable_mask:
                img = np.array(_img)
                rgb, alpha = img[:, :, :3], img[:, :, 3:]
                fp_alpha = alpha / 255
                rgb[:] = rgb * fp_alpha + (255 - alpha)
                img = rgb
            else:
                npimg = np.array(_img)
                img, mask = npimg[:, :, :3], npimg[:, :, 3]
        else:
            img = np.array(_img.convert("RGB"))
        img, mask = self.augment(img, mask, index)
        h,w = img.shape[:2]
        scale_ratio = self.base_len/min(h,w)
        if scale_ratio<1:
            img = cv2.resize(img, (round(w*scale_ratio), round(h*scale_ratio)), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (round(w*scale_ratio), round(h*scale_ratio)), interpolation=cv2.INTER_AREA)
        elif scale_ratio>1:
            img = cv2.resize(img, (round(w*scale_ratio), round(h*scale_ratio)), interpolation=cv2.INTER_CUBIC)
            if mask is not None:
                mask = cv2.resize(mask, (round(w*scale_ratio), round(h*scale_ratio)), interpolation=cv2.INTER_CUBIC)
            print(f"\033[33mWarning: {p}'s shorter side is {min(h,w)}px after rotation, which is smaller than base_len {self.base_len}px.\033[0m")
        return False, img, mask, prompt

    def fix_aspect_randomness(self, rng:np.random.Generator):
        if not self.kwargs.get('augmentation') or not self.kwargs['augmentation'].get('rotate'):
            self.curr_res = self.raw_res
            return
        
        l,h = self.kwargs['augmentation']['rotate']
        self.rand_list:list[float] = rng.uniform(l,h,size=self.img_length).tolist()
        self.curr_res = [
            rotatedRectWithMaxArea(*self.raw_res[i], self.rand_list[i]*np.pi/180) for i in range(self.img_length)
        ]
        
class AspectRatioDataset(Dataset):
    def __init__(
        self,
        batch_size:int,
        root_path:Path|str,
        tokenizer:CLIPTokenizer,
        ucg=0,
        rank=0,
        prompt_processor:str|None=None,
        enable_mask=False,
        dtype=torch.float16,
        use_central_crop=False,
        base_len=512,
        max_len=1024,
        augmentation:Optional[dict[str, Any]]=None,
        seed = 114514,
        **kwargs,
    ):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        root_path = Path(root_path)
        assert root_path.exists()
        if root_path.is_file():
            assert root_path.suffix == ".h5"
            store_class = LatentStore
        else:
            assert root_path.is_dir()
            store_class = DirectoryImageStore
        self.store = store_class(root_path, rank=rank, enable_mask=enable_mask, ucg=ucg, prompt_processor=prompt_processor, use_central_crop=use_central_crop,tokenizer=tokenizer, dtype=dtype, augmentation=augmentation, base_len=base_len, **kwargs)
        
        self.base_len = base_len
        self.max_len = max_len
        self._length = int(np.ceil(self.store.img_length/self.batch_size))
        self.generate_buckets()
        self.first_time = True
        
    def __len__(self):
        return self._length
    
    def generate_buckets(self):
        b = np.arange(self.base_len, self.max_len+1, 64)
        B = np.concatenate([[self.base_len]*(len(b)-1), b])
        self.buckets_sizes = np.stack((B,B[::-1]), axis=1)
        self.bucket_ratios = self.buckets_sizes[:,0]/self.buckets_sizes[:,1]
        
    def assign_buckets(self):
        
        self.store.fix_aspect_randomness(self.rng)
        img_curr_res = np.array(self.store.curr_res)
        img_ratios = img_curr_res[:,0]/img_curr_res[:,1]
        img_idxs = np.argsort(img_ratios)
        landscape_idxs = img_idxs[img_ratios[img_idxs] <= 1]
        portrait_idxs = img_idxs[img_ratios[img_idxs] > 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        
        # Initial assignment, images are rounded towards the base bucket
        bucket_idx = 0
        
        self.store.to_ratio = np.empty(self.store.img_length)
        idx_size=landscape_idxs.size
        reminder = idx_size%self.batch_size
        
        it = []
        if reminder:    
            if idx_size >= self.batch_size:
                it = np.split(landscape_idxs[:-reminder], idx_size//self.batch_size)
            it.append(landscape_idxs[-reminder:])
        else:
            if idx_size >= self.batch_size:
                it = np.split(landscape_idxs, idx_size//self.batch_size)
        
        for idx_chunk in it:
            idx = idx_chunk[-1]
            while self.bucket_ratios[bucket_idx]<img_ratios[idx]:
                bucket_idx += 1
            self.bucket_content[bucket_idx].extend(idx_chunk)
            self.store.to_ratio[idx_chunk] = self.bucket_ratios[bucket_idx]
            
        
        idx_size=portrait_idxs.size
        reminder = idx_size%self.batch_size
        
        it = []
        if reminder:
            if idx_size >= self.batch_size:
                it = np.split(portrait_idxs[reminder:], idx_size//self.batch_size)[::-1]
            it.append(portrait_idxs[:reminder])
        else:
            if idx_size >= self.batch_size:
                it = np.split(portrait_idxs, idx_size//self.batch_size)[::-1]

        bucket_idx = len(self.buckets_sizes)-1
        for idx_chunk in it:
            idx = idx_chunk[0]
            while self.bucket_ratios[bucket_idx]>img_ratios[idx]:
                bucket_idx -= 1
            self.bucket_content[bucket_idx].extend(idx_chunk)
            self.store.to_ratio[idx_chunk] = self.bucket_ratios[bucket_idx]
            
    def assign_batches(self):
        self.batch_idxs = []
        for bucket in self.bucket_content:
            if not bucket:
                continue
            reminder = len(bucket)%self.batch_size
            bucket = np.array(bucket)
            self.rng.shuffle(bucket)
            if not reminder:
                self.batch_idxs.extend(bucket.reshape(-1, self.batch_size))
            else:
                self.batch_idxs.extend(bucket[:-reminder].reshape(-1, self.batch_size))
                self.batch_idxs.append(bucket[-reminder:])
        np.random.shuffle(self.batch_idxs)
        
    def put_most_oom_like_batch_first(self):
        i=0
        while True:
            b_landscape, b_portrait = self.bucket_content[i], self.bucket_content[-i-1]
            if b_landscape:
                idx = b_landscape[0]
                break
            elif b_portrait:
                idx = b_portrait[0]
                break
            i+=1
        for i, batch_idxs in enumerate(self.batch_idxs):
            if idx in batch_idxs:
                self.batch_idxs[0], self.batch_idxs[i] = self.batch_idxs[i], self.batch_idxs[0]
                break
 
    def __getitem__(self, idx):
        if self.first_time:
            self.store.fix_aspect_randomness(self.rng)
            self.assign_buckets()
            self.assign_batches()
            self.first_time = False
            self.put_most_oom_like_batch_first()
        img_idxs = self.batch_idxs[idx]
        return self.store.get_batch(img_idxs)

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset:AspectRatioDataset = worker_info.dataset # type: ignore
    random.seed(worker_info.seed)
    dataset.store.fix_aspect_randomness(dataset.rng)
    dataset.assign_buckets()
    dataset.assign_batches()
    if dataset.first_time:
        dataset.put_most_oom_like_batch_first()
        dataset.first_time = False
    