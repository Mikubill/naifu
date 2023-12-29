import argparse
import cv2
import hashlib
import h5py as h5
import json
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Callable, Generator, Optional


def load_entry(p: Path, label_ext: str = ".txt"):
    _img = Image.open(p)
    with p.with_suffix(label_ext).open("r") as f:
        prompt = f.read()
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
    def __init__(self, root: str | Path, dtype=torch.float32):
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
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            w, h = Image.open(p).size
            self.raw_res.append((h, w))
        self.length = len(self.raw_res)
        print(f"Loaded {self.length} image sizes")
        
        self.target_area = 1024 * 1024
        self.max_size, self.min_size, self.divisible = 2048, 512, 64
        self.generate_buckets()
        self.assign_buckets()

    def generate_buckets(self):
        assert (
            self.target_area % 4096 == 0
        ), "target area (h * w) must be divisible by 64"
        width = np.arange(self.min_size, self.max_size + 1, self.divisible)
        height = np.minimum(self.max_size, ((self.target_area // width) // self.divisible) * self.divisible,)
        valid_mask = height >= self.min_size

        resos = set(zip(width[valid_mask], height[valid_mask]))
        resos.update(zip(height[valid_mask], width[valid_mask]))
        resos.add(((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2)
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)}

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
        
    @torch.no_grad()
    def fit_bucket(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[-2:]
        base_ratio = h / w
        target_ratio = self.to_ratio[idx]
        target_h, target_w = self.ratio_to_bucket[target_ratio]
        if base_ratio > target_ratio:
            resize_h = int(target_w * base_ratio)
            resize_w = int(target_w)
        else:
            resize_h = int(target_h)
            resize_w = int(target_h / base_ratio)
        img = transforms.Resize((resize_h, resize_w), antialias=True)(img)
        return img

    def __getitem__(self, index) -> tuple[list[torch.Tensor], str, str, (int, int)]:
        img, prompt = load_entry(self.paths[index])
        original_size = img.shape[:2]
        img = self.fit_bucket(index, self.tr(img)).to(self.dtype)
        sha1 = get_sha1(self.paths[index])
        return img, self.paths[index], prompt, sha1, original_size

    def __len__(self):
        return len(self.paths)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="root directory of images"
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="output file")
    parser.add_argument("--dtype", "-d", type=str, default="float32", help="data type")
    parser.add_argument(
        "--num_workers", "-n", type=int, default=4, help="number of dataloader workers"
    )
    parser.add_argument("--device", "-D", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--slice-vae", "-s", action="store_true", help="slice vae, saves some vram"
    )
    parser.add_argument(
        "--tile-vae", "-t", action="store_true", help="tile vae, saves a lot of vram"
    )
    parser.add_argument(
        "--compress",
        "-c",
        type=str,
        default=None,
        help="compression algorithm for output hdf5 file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.input
    opt = Path(args.output)
    dtype = getattr(torch, args.dtype)
    num_workers = args.num_workers
    device = args.device
    vae_path = "stabilityai/sdxl-vae"

    dataset = LatentEncodingDataset(root, dtype=dtype)
    dataloader = DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=num_workers
    )
    vae = AutoencoderKL.from_pretrained(vae_path, repo_type="datasets").to(
        device=device, dtype=dtype
    )
    if args.slice_vae:
        vae.enable_slicing()
    if args.tile_vae:
        vae.enable_tiling()
    vae = vae.to(device)

    print(f"Starting encoding...")
    if not opt.exists():
        opt.mkdir()

    assert opt.is_dir(), f"{opt} is not a directory"
    h5_cache_file = opt / "cache.h5"
    file_mode = "w" if not h5_cache_file.exists() else "r+"
    dataset_mapping = {}
    with h5.File(h5_cache_file, file_mode, libver="latest") as f:
        with torch.no_grad():
            for i, (img, basepath, prompt, sha1, original_size) in enumerate(
                tqdm(dataloader)
            ):
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
                        f"\033[33mWarning: {sha1} is already cached. Skipping... \033[0m"
                    )
                    continue
                    
                latent = vae.encode(img.unsqueeze(0).cuda(), return_dict=False)[0]
                latent.deterministic = True
                latent = latent.sample()[0]
                d = f.create_dataset(
                    f"{sha1}.latents",
                    data=latent.half().cpu().numpy(),
                    compression=args.compress,
                )
                d.attrs["scale"] = False

    with open(opt / "dataset.json", "w") as f:
        json.dump(dataset_mapping, f, indent=4)
