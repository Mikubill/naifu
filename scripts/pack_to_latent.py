import argparse
from torchvision import transforms
import torch
from diffusers import AutoencoderKL
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision.transforms import Normalize
import h5py as h5
import re
import hashlib
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Callable, Generator, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
import h5py as h5

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
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(hr), int(wr)


def load_entry(p:Path, enable_mask = True, label_ext:str='.label'):
    _img = Image.open(p)
    with p.with_suffix(label_ext).open('r') as f:
        prompt = f.read()
    mask = None
    if _img.mode == "RGB":
        img = np.array(_img)
    elif _img.mode == "RGBA":
        if not enable_mask:
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
    return img, prompt, mask


def rotate_and_crop(image:np.ndarray, mask:None|np.ndarray, angles:list[float]) -> tuple[list[np.ndarray], list[np.ndarray|None]]:
    images, masks = [], []
    for angle in angles:
        H,W = image.shape[:2]
        h,w = rotatedRectWithMaxArea(H,W,np.deg2rad(angle))
        rot = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
        dw, dh = (W-w)//2, (H-h)//2
        _image = cv2.warpAffine(image, rot, (W,H))[dh:dh+h, dw:dw+w]
        _mask = None
        if mask is not None:
            _mask = cv2.warpAffine(mask, rot, (W,H))[dh:dh+h, dw:dw+w]
        images.append(_image)
        masks.append(_mask)
    return images, masks


def flip_entry(image:np.ndarray, mask:None|np.ndarray, flip:list[int]|np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray|None]]:
    images, masks = [], []
    for f in flip:
        _image = image
        _mask = None
        if f == -1:
            _image = cv2.flip(image, 1)
            if mask is not None:
                _mask = cv2.flip(mask, 1)
        images.append(_image)
        masks.append(_mask)
    return images, masks

def fit_base_size(img:np.ndarray,mask:None|np.ndarray, base_size:int) -> tuple[np.ndarray, None|np.ndarray]:
    H,W = img.shape[:2]
    r = base_size/min(H,W)
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    img = cv2.resize(img, (round(W*r), round(H*r)), interpolation=interp)
    if mask is not None:
        mask = cv2.resize(mask, (round(W*r), round(H*r)), interpolation=interp)
    assert min(img.shape[:2]) == base_size
    if max(img.shape[:2])%8==0:
        return img, mask
    H,W = img.shape[:2]
    h,w = int(H/8)*8, int(W/8)*8
    dh, dw = (H-h)//2, (W-w)//2
    img = img[dh:dh+h, dw:dw+w]
    if mask is not None:
        mask = mask[dh:dh+h, dw:dw+w]
    return img, mask


def gen_aug_group(img:np.ndarray, mask:np.ndarray|None, rotate:list[float]|np.ndarray, flip:list[int]|np.ndarray, base_size:int):
    img_list, mask_list = [], []
    for r in rotate:
        img_list_r, mask_list_r = rotate_and_crop(img, mask, [r])
        for img_r, mask_r in zip(img_list_r, mask_list_r):
            img_rf_list, mask_rf_list = flip_entry(img_r, mask_r, flip)
            img_list.extend(img_rf_list)
            mask_list.extend(mask_rf_list)
    return [fit_base_size(img, mask, base_size) for img, mask in zip(img_list, mask_list)]


md5_m = re.compile(r'(?<![a-zA-Z0-9])([0-9a-fA-F]{32})(?![a-zA-Z0-9])')
def get_md5(path: Path):
    m = md5_m.search(path.name)
    if m:
        return m.group(1).lower()
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

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
                
                    
class LatentEncodingDataset(Dataset):
    def __init__(self,
                 root:str|Path,
                 rotate:list[float]|np.ndarray,
                 p_rotate:list[float]|np.ndarray,
                 flip:list[int]|np.ndarray,
                 p_flip:list[float]|np.ndarray,
                 base_size:int=1152,
                 dtype = torch.float32,
                 enable_mask = True,
                 label_ext = '.label'
                 ):
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.paths = sorted(list(dirwalk(Path(root), is_img)))
        print(f'Found {len(self.paths)} images')
        self.rotate = rotate
        self.p_rotate = p_rotate
        self.flip = flip
        self.p_flip = p_flip
        self.base_size = base_size
        self.dtype = dtype
        self.enable_mask = enable_mask
        self.label_ext = label_ext
    
    def __getitem__(self, index) -> tuple[list[torch.Tensor], list[torch.Tensor|None], str]:
        img, prompt, mask = load_entry(self.paths[index], enable_mask = self.enable_mask, label_ext=self.label_ext)
        md5 = get_md5(self.paths[index])
        gp = gen_aug_group(img, mask, self.rotate, self.flip, self.base_size)
        imgs, masks = zip(*gp)
        torch_imgs = [self.tr(img).to(self.dtype) for img in imgs] # type: ignore
        del imgs
        torch_masks = []
        for mask in masks:
            if mask is None:
                torch_masks.append(None)
                continue
            h,w = mask.shape
            mask = mask.reshape(h//8, 8, w//8, 8).min(axis=(1,3))/255
            torch_masks.append(torch.from_numpy(mask).to(self.dtype))
        del masks
        return torch_imgs, torch_masks, prompt, md5 # type: ignore       
    
    def __len__(self):
        return len(self.paths)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, required=True, help='root directory of images')
    parser.add_argument('--vae-path', '-vae', type=str, required=True, help='path to vae model')
    parser.add_argument('--base_size', '-b', type=int, default=512, help='base size')
    parser.add_argument('--enable_mask', '-m', action='store_true', help='enable mask')
    parser.add_argument('--label_ext', '-l', type=str, default='.label', help='label extension')
    parser.add_argument('--dtype', '-d', type=str, default='float16', help='data type')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file')
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--rotate', '-R', type=float, nargs='+', default=[-3., 0., 3.], help='rotate angles')
    parser.add_argument('--p_rotate', '-pR', type=float, nargs='+', default=[0.25, 0.5, 0.25], help='rotate angles')
    parser.add_argument('--flip', '-F', type=int, nargs='+', default=[-1, 1], help='flip')
    parser.add_argument('--p_flip', '-pF', type=float, nargs='+', default=[0.5, 0.5], help='flip')
    parser.add_argument('--device', '-D', type=str, default='cuda:0', help='device')
    parser.add_argument('--slice-vae', '-s', action='store_true', help='slice vae, saves some vram')
    parser.add_argument('--tile-vae', '-t', action='store_true', help='tile vae, saves a lot of vram')
    parser.add_argument('--compress', '-c', type=str, default=None, help='compression algorithm for output hdf5 file')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    
    args = get_args()
    root = args.root
    rotate = args.rotate
    p_rotate = args.p_rotate
    flip = args.flip
    p_flip = args.p_flip
    base_size = args.base_size
    dtype = getattr(torch, args.dtype)
    num_workers = args.num_workers
    p_flip = np.array(p_flip)/sum(p_flip)
    p_rotate = np.array(p_rotate)/sum(p_rotate)
    device = args.device
    
    assert len(rotate) == len(p_rotate), 'rotate and p_rotate must have the same length'
    assert len(flip) == len(p_flip), 'flip and p_flip must have the same length'
        
    dataset = LatentEncodingDataset(root, rotate, p_rotate, flip, p_flip, base_size=base_size, dtype=dtype, label_ext=args.label_ext)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=num_workers)
    
    
    print(f'Loading vae... need {len(flip)*len(rotate)} forward passes per image for augmentation')
    vae:AutoencoderKL = AutoencoderKL.from_pretrained(args.vae_path).to(device=device, dtype=dtype) # type: ignore
    if args.slice_vae:
        vae.enable_slicing()
    if args.tile_vae:
        vae.enable_tiling()
    vae = vae.to(device)
    
    print(f'Starting encoding...')
    with h5.File(args.output, 'w') as f:
        with torch.no_grad():
            for i, (imgs, masks, prompt, md5) in enumerate(tqdm(dataloader)):
                if md5 in f:
                    print(f"\033[33mWarning: {md5} is already cached. Skipping... \033[0m")
                    continue
                g = f.create_group(md5)
                g.attrs['prompt'] = prompt
                ps = np.outer(p_flip,p_rotate).reshape(-1)  
                g.create_dataset('probs', data=ps, compression=args.compress)
                for j, img in enumerate(imgs):
                    latent = vae.encode(img.unsqueeze(0).cuda(), return_dict=False)[0]
                    latent.deterministic = True
                    latent = latent.sample()[0]
                    
                    g.create_dataset(f'{j}.pixel', data=latent.half().cpu().numpy(), compression=args.compress)
                    mask = masks[j]
                    if mask is not None:
                        g.create_dataset(f'{j}.mask', data=masks[j].half().cpu().numpy(), compression=args.compress)
            
