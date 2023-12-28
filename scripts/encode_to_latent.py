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


def fit_base_size(img: np.ndarray, base_size: int) -> tuple[np.ndarray, None | np.ndarray]:
    H, W = img.shape[:2]
    r = base_size / min(H, W)
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (round(W * r), round(H * r)), interpolation=interp)
    assert min(img.shape[:2]) == base_size
    if max(img.shape[:2]) % 8 == 0:
        return img
    H, W = img.shape[:2]
    h, w = int(H / 8) * 8, int(W / 8) * 8
    dh, dw = (H - h) // 2, (W - w) // 2
    img = img[dh : dh + h, dw : dw + w]
    return img

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
    def __init__(
        self,
        root: str | Path,
        base_size: int = 1152,
        dtype=torch.float32,
    ):
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.paths = sorted(list(dirwalk(Path(root), is_img)))
        print(f"Found {len(self.paths)} images")
        self.base_size = base_size
        self.dtype = dtype

    def __getitem__(self, index) -> tuple[list[torch.Tensor], str, str, (int, int)]:
        img, prompt = load_entry(self.paths[index])
        original_size = img.shape[:2]
        img = self.tr(fit_base_size(img, base_size)).to(self.dtype)
        sha1 = get_sha1(self.paths[index])
        return img, self.paths[index], prompt, sha1, original_size  

    def __len__(self):
        return len(self.paths)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="root directory of images")
    parser.add_argument("--output", "-o", type=str, required=True, help="output file")
    parser.add_argument("--base_size", "-b", type=int, default=1024, help="base size")
    parser.add_argument("--dtype", "-d", type=str, default="float32", help="data type")
    parser.add_argument("--num_workers", "-n", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--device", "-D", type=str, default="cuda:0", help="device")
    parser.add_argument("--slice-vae", "-s", action="store_true", help="slice vae, saves some vram")
    parser.add_argument( "--tile-vae", "-t", action="store_true", help="tile vae, saves a lot of vram")
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
    base_size = args.base_size
    opt = Path(args.output)
    dtype = getattr(torch, args.dtype)
    num_workers = args.num_workers
    device = args.device
    vae_path = "stabilityai/sdxl-vae"

    dataset = LatentEncodingDataset(root, base_size=base_size, dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=num_workers)
    vae = AutoencoderKL.from_pretrained(vae_path, repo_type="datasets").to(device=device, dtype=dtype)  
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
            for i, (img, basepath, prompt, sha1, original_size) in enumerate(tqdm(dataloader)):
                h, w = original_size
                dataset_mapping[sha1] = {
                    "train_use": True,
                    "train_caption": prompt,
                    "file_path": str(basepath),
                    "train_width": h,
                    "train_height": w
                }
                if f"{sha1}.latents" in f:
                    print(f"\033[33mWarning: {sha1} is already cached. Skipping... \033[0m")
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
