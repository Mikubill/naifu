import argparse
import functools
import hashlib
import math
import cv2
import h5py as h5
import json
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Generator, Optional


def load_entry(p: Path, label_ext: str = ".txt"):
    _img = Image.open(p)
    with p.with_suffix(label_ext).open("r") as f:
        prompt = f.read()
    if _img.mode == "RGB":
        img = np.array(_img)
    elif _img.mode == "RGBA":
        # transparent images
        baimg = Image.new('RGB', _img.size, (255, 255, 255))
        baimg.paste(_img, (0, 0), _img)
        img = np.array(baimg)
    else:
        img = np.array(_img.convert("RGB"))
    return img, prompt


def get_sha1(path: Path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])


def is_img(path: Path):
    return path.suffix in image_suffix


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class LatentEncodingDataset(Dataset):
    def __init__(self, root: str | Path, dtype=torch.float32, no_upscale=False):
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.paths = sorted(list(dirwalk(Path(root), is_img)))
        print(f"Found {len(self.paths)} images")
        self.dtype = dtype
        self.raw_res = []

        remove_paths = []
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                w, h = Image.open(p).size
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mSkipped: error processing {p}: {e}\033[0m")
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)
        print(f"Loaded {self.length} image sizes")
        
        self.fit_bucket_func = self.fit_bucket
        if no_upscale:
            self.fit_bucket_func = self.fit_bucket_no_upscale

        self.target_area = 1024 * 1024
        self.max_size, self.min_size, self.divisible = 2048, 512, 64
        self.generate_buckets()
        self.assign_buckets()

    def generate_buckets(self):
        assert (
            self.target_area % 4096 == 0
        ), "target area (h * w) must be divisible by 64"
        width = np.arange(self.min_size, self.max_size + 1, self.divisible)
        height = np.minimum(
            self.max_size,
            ((self.target_area // width) // self.divisible) * self.divisible,
        )
        valid_mask = height >= self.min_size

        resos = set(zip(width[valid_mask], height[valid_mask]))
        resos.update(zip(height[valid_mask], width[valid_mask]))
        resos.add(
            ((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2
        )
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {
            ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)
        }

    def assign_buckets(self):
        img_res = np.array(self.raw_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(self.bucket_ratios - img_ratio)
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]

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

    @torch.no_grad()
    def fit_bucket(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        base_ratio = h / w
        target_ratio = self.to_ratio[idx]
        target_h, target_w = self.ratio_to_bucket[target_ratio]
        resize_h, resize_w = self.fit_dimensions(base_ratio, target_h, target_w)
        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_CUBIC
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    @torch.no_grad()
    def fit_bucket_no_upscale(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        img_area = h * w

        # Check if the image needs to be resized (i.e., only allow downsizing)
        if img_area > self.target_area:
            scale_factor = math.sqrt(self.target_area / img_area)
            resize_w = math.floor(w * scale_factor / self.divisible) * self.divisible
            resize_h = math.floor(h * scale_factor / self.divisible) * self.divisible
        else:
            resize_w, resize_h = w, h

        target_w = resize_w - resize_w % self.divisible
        target_h = resize_h - resize_h % self.divisible
            
        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_CUBIC
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    def __getitem__(self, index) -> tuple[list[torch.Tensor], str, str, (int, int)]:
        try:
            img, prompt = load_entry(self.paths[index])
            original_size = img.shape[:2]
            img, dhdw = self.fit_bucket_func(index, img) 
            img = self.tr(img).to(self.dtype)
            sha1 = get_sha1(self.paths[index])
        except Exception as e:
            print(f"\033[31mError processing {self.paths[index]}: {e}\033[0m")
            return None, self.paths[index], None, None, None, None
        return img, self.paths[index], prompt, sha1, original_size, dhdw

    def __len__(self):
        return len(self.paths)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="root directory of images"
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="output file")
    parser.add_argument("--no-upscale", "-nu", action="store_true", help="do not upscale images")
    parser.add_argument("--dtype", "-d", type=str, default="float32", help="data type")
    parser.add_argument("--num_workers", "-n", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--model", "-m", type=str, default="stabilityai/sdxl-vae", help="model path")
    parser.add_argument("--subfolder", type=str, default=None, help="use subfolder to locate vae")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.input
    opt = Path(args.output)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    num_workers = args.num_workers

    vae = AutoencoderKL.from_pretrained(args.model, subfolder=args.subfolder).to(dtype)
    vae.requires_grad_(False)
    vae.eval().cuda()

    dataset = LatentEncodingDataset(root, dtype=dtype, no_upscale=args.no_upscale)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    opt.mkdir(exist_ok=True, parents=True)
    assert opt.is_dir(), f"{opt} is not a directory"

    cache_filename = "cache.h5"
    dataset_mapping = {}

    h5_cache_file = opt / cache_filename
    print(f"Saving cache to {h5_cache_file}")
    file_mode = "w" if not h5_cache_file.exists() else "r+"

    with h5.File(h5_cache_file, file_mode, libver="latest") as f:
        with torch.no_grad():
            for img, basepath, prompt, sha1, original_size, dhdw in tqdm(dataloader):
                if sha1 is None:
                    print(
                        f"\033[33mWarning: {basepath} is invalid. Skipping... \033[0m"
                    )
                    continue

                h, w = original_size
                dataset_mapping[sha1] = {
                    "train_use": True,
                    "train_caption": prompt,
                    "file_path": str(basepath),
                    "train_width": w,
                    "train_height": h,
                }
                if f"{sha1}.latents" in f:
                    print(
                        f"\033[33mWarning: {str(basepath)} is already cached. Skipping... \033[0m"
                    )
                    continue

                img = img.unsqueeze(0).cuda()
                latent = vae.encode(img, return_dict=False)[0]
                latent.deterministic = True
                latent = latent.sample()[0]
                d = f.create_dataset(
                    f"{sha1}.latents",
                    data=latent.float().cpu().numpy(),
                    compression="gzip",
                )
                d.attrs["scale"] = False
                d.attrs["dhdw"] = dhdw

    with open(opt / "dataset.json", "w") as f:
        json.dump(dataset_mapping, f, indent=4)
