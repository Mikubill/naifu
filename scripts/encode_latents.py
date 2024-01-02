import argparse
import functools
import hashlib
import math
import cv2
import h5py as h5
import json
import numpy as np
import torch
import torch.distributed as dist

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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

        remove_paths = []
        for p in tqdm(
            self.paths, desc="Loading image sizes", leave=False, ascii=True,
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
        resos.add(((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2)
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
        img = img[dh:dh+target_h, dw:dw+target_w]
        return img, (dh, dw)

    def __getitem__(self, index) -> tuple[list[torch.Tensor], str, str, (int, int)]:
        try:
            img, prompt = load_entry(self.paths[index])
            original_size = img.shape[:2]
            img, dhdw = self.fit_bucket(index, img)
            img = self.tr(img).to(self.dtype)
            sha1 = get_sha1(self.paths[index])
        except Exception as e:
            print(f"\033[31mError processing {self.paths[index]}: {e}\033[0m")
            return None, self.paths[index], None, None, None
        return img, self.paths[index], prompt, sha1, original_size, dhdw

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
        default='gzip',
        help="compression algorithm for output hdf5 file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    multi_node = torch.cuda.device_count() > 1
    rank = 0
    if multi_node:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()

    root = args.input
    opt = Path(args.output)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    num_workers = args.num_workers

    vae_path = "stabilityai/sdxl-vae"
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    if args.slice_vae: vae.enable_slicing()
    if args.tile_vae: vae.enable_tiling()
    
    dataset = LatentEncodingDataset(root, dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    opt.mkdir(exist_ok=True, parents=True)
    assert opt.is_dir(), f"{opt} is not a directory"

    cache_filename = "cache.h5" 
    dataset_mapping = {}
    vae.to(torch.device(f"cuda:{rank}"))
    
    if multi_node:
        cache_filename = f"cache_{rank+1}.h5" 
        h5_cache_file = opt / cache_filename
        sampler = DistributedSampler(dataset, rank=rank)
        dataloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers)
    
    h5_cache_file = opt / cache_filename
    print(f"Saving cache to {h5_cache_file}")
    file_mode = "w" if not h5_cache_file.exists() else "r+"
    with h5.File(h5_cache_file, file_mode, libver="latest") as f:
        with torch.no_grad():
            for img, basepath, prompt, sha1, original_size, dhdw in tqdm(dataloader, position=rank):
                if sha1 is None:
                    print(f"\033[33mWarning: {basepath} is invalid. Skipping... \033[0m")
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
                    print(f"\033[33mWarning: {sha1} is already cached. Skipping... \033[0m")
                    continue
                
                img = img.unsqueeze(0).to(torch.device(f"cuda:{rank}"))
                latent = vae.encode(img, return_dict=False)[0]
                latent.deterministic = True
                latent = latent.sample()[0]
                d = f.create_dataset(
                    f"{sha1}.latents",
                    data=latent.half().cpu().numpy(),
                    compression=args.compress,
                )
                d.attrs["scale"] = False
                d.attrs["dhdw"] = dhdw

    json_filename = f"dataset_{rank+1}.json" if multi_node else "dataset.json"
    with open(opt / json_filename, "w") as f:
        json.dump(dataset_mapping, f, indent=4)

    if multi_node:
        dist.barrier()
        if rank == 0:
            # merge all json and h5 file
            with open(opt / "dataset.json", "w") as f:
                for i in range(dist.get_world_size()):
                    with open(opt / f"dataset_{i+1}.json", "r") as f1:
                        f.write(f1.read())
                    Path(opt / f"dataset_{i+1}.json").unlink()

            with h5.File(opt / "cache.h5", "w", libver="latest") as f:
                for i in range(dist.get_world_size()):
                    with h5.File(opt / f"cache_{i+1}.h5", "r", libver="latest") as f1:
                        for k in f1.keys():
                            if k not in f:
                                f.copy(f1[k], k)
                    Path(opt / f"cache_{i+1}.h5").unlink()
        dist.barrier()
        dist.destroy_process_group()
