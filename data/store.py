import hashlib
import os
import random
import torch
import h5py
import os, binascii
import re, json
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from data.buckets import AspectRatioBucket
from lib.utils import rank_zero_print, get_local_rank, get_world_size

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

class ImageStore(torch.utils.data.IterableDataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        img_path,
        size=512,
        center_crop=False,
        ucg=0,
        rank=0,
        tag_processor=[],
        allow_duplicates=False,
        **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.ucg = ucg
        self.rank = get_local_rank()
        self.world_size = get_world_size()
        self.dataset = img_path
        self.use_latent_cache = False
        self.allow_duplicates = allow_duplicates
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        if isinstance(self.dataset, str):
            self.dataset = [self.dataset]
            
        if isinstance(tag_processor, str):
            tag_processor = [tag_processor]
            
        self.tag_processor = []
        for processor in tag_processor:
            if processor != "":
                self.tag_processor.append(get_class(processor))
        
        self.latent_cache = {}
        self.update_store()

    def prompt_resolver(self, x: str):
        img_item = (x, "")
        fp = os.path.splitext(x)[0]

        try:
            with open(fp + ".txt") as f:
                content = f.read()
                new_prompt = content
        except FileNotFoundError:
            rank_zero_print(f"Prompt file not found for {x}")
            new_prompt = ""

        return str(x), new_prompt

    def update_store(self):   
        self.entries = []
        bar = tqdm(total=-1, desc=f"Loading images", disable=self.rank not in [0, -1])
        folders = []
        for entry in self.dataset:
            if self.allow_duplicates and not isinstance(entry, str):
                folders.extend([entry[0] for _ in range(entry[1])])
            else:
                folders.append(entry)
        
        for entry in folders:       
            if Path(entry).suffix == ".json":
                # load json and extract entries
                # json format: {"<image_sha1_hash>": {"train_use": true, "train_caption": "", "train_width": 1024, "train_height": 1024}}
                with open(entry, 'r') as f:
                    data = json.load(f)
                
                for img_hash, img_info in data.items():
                    if not img_info["train_use"]:
                        continue
                    prompt = img_info["train_caption"]
                    _, skip = self.process_tags(prompt)
                    if skip:
                        continue
                    size = (img_info["train_width"], img_info["train_height"])
                    self.entries.append((img_hash, prompt, size))
                    bar.update(1)
                continue
            
            for x in Path(entry).rglob("*"):
                if not (x.is_file() and x.suffix in [".jpg", ".png", ".webp", ".bmp", ".gif", ".jpeg", ".tiff"]):
                    continue
                img, prompt = self.prompt_resolver(x)
                _, skip = self.process_tags(prompt)
                if skip:
                    continue
                
                if self.allow_duplicates:
                    prefix = binascii.hexlify(os.urandom(5))
                    img = f"{prefix}@{img}"
                    
                image_size = self.read_img(x).size
                self.entries.append((img, prompt, image_size))
                bar.update(1)

        self._length = len(self.entries)
        random.shuffle(self.entries)
        
    def cache_latents(self, vae, config=None):
        self.use_latent_cache = True
        self.latents_cache = {}
        for entry in tqdm(self.entries, desc=f"Caching latents", disable=self.rank not in [0, -1]):
            image = torch.stack([self.image_transforms(self.read_img(entry[0]))])
            latent = vae(image.to(vae.device, dtype=vae.dtype))
            self.latents_cache[entry[0]] = latent.detach().squeeze(0).cpu()
            
        return False

    def read_img(self, filepath):  
        if self.allow_duplicates and "@" in filepath:
            filepath = filepath[filepath.index("@")+1:]    
        img = Image.open(filepath)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img

    def process_tags(self, tags):
        if len(self.tag_processor) == 0:
            return tags, False
        
        reject = False
        for processor in self.tag_processor:
            tags, rej_cur = processor(tags)
            reject = reject or rej_cur
            
        return tags, reject
        
    # cc: openai
    def crop_align(self, arrays):
        min_width = np.min([array.shape[2] for array in arrays])
        min_height = np.min([array.shape[1] for array in arrays])

        start_width = []
        start_height = []

        for array in arrays:
            width_diff = array.shape[2] - min_width
            height_diff = array.shape[1] - min_height
            start_width.append(width_diff // 2)
            start_height.append(height_diff // 2)

        end_width = [start + min_width for start in start_width]
        end_height = [start + min_height for start in start_height]
        
        cropped_arrays = [
            array[:, start_height[i]:end_height[i], start_width[i]:end_width[i]]
            for i, array in enumerate(arrays)
        ]
        return cropped_arrays

    def collate_fn(self, examples):
        pixel_values = self.crop_align([example["images"] for example in examples])
        result = {
            "pixel_values": torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
            "prompts": [example["prompts"] for example in examples],
            "target_size_as_tuple": [example["target_size_as_tuple"] for example in examples],
            "original_size_as_tuple": [example["original_size_as_tuple"] for example in examples],
            "crop_coords_top_left": [example["crop_coords_top_left"] for example in examples],
        }
        return result

    def __len__(self):
        return self._length // self.world_size

    def __iter__(self):
        local_rank = self.rank if self.rank != -1 else 0
        for entry in self.entries[local_rank::self.world_size]:
            example = {}
            instance_path, instance_prompt, size = entry
            
            if self.use_latent_cache:
                example["images"] = self.latents_cache[instance_path]
            else:
                instance_image = self.read_img(instance_path)
                example["images"] = self.image_transforms(instance_image)
            
            w, h  = size
            example["prompts"] = instance_prompt
            example["target_size_as_tuple"] = (h, w)
            example["original_size_as_tuple"] = (h, w)
            example["crop_coords_top_left"] = (0, 0)
            yield example


class AspectRatioDataset(ImageStore):
    def __init__(self, arb_config, debug_arb=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug_arb
        self.prompt_cache = {}
        self.kwargs = kwargs
        self.cache_enabled = kwargs.get("cache_latents", False)
        self.cache_dir = kwargs.get("cache_dir", "cache")
        self.cache_bsz = kwargs.get("cache_bsz", 4)
        self.use_legacy_key = kwargs.get("use_legacy_key", False)
        self.hashes = {}

        self.arb_config = arb_config
        if arb_config["debug"]:
            rank_zero_print(f"BucketManager initialized using config: {self.arb_config}")
        else:
            rank_zero_print(f"BucketManager initialized with base_res = {self.arb_config['base_res']}, max_size = {self.arb_config['max_size']}")
        
        for path, prompt, size in self.entries:
            self.prompt_cache[path] = prompt
        
        if self.cache_enabled:
            self.hash_all()
            
        self.init_buckets()
        
    def init_buckets(self):
        id_size_map = {fp: size for fp, prompt, size in self.entries}
        self.buckets = AspectRatioBucket(id_size_map, **self.arb_config)

    def denormalize(self, img, mean=0.5, std=0.5):
        res = transforms.Normalize((-1*mean/std), (1.0/std))(img.squeeze(0))
        res = torch.clamp(res, 0, 1)
        return res

    def transformer(self, img, size, center_crop=False):
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

        new_img = image_transforms(img)
        return new_img

    def hash_all(self):
        if self.use_legacy_key:
            return
         
        sha1_pattern = re.compile(r"^[a-fA-F0-9]{40}$")
        store = {}
        for filepath, caption, size in tqdm(self.entries, desc="Hashing files", disable=self.rank not in [0, -1]):
            if self.allow_duplicates and "@" in filepath:
                filepath = filepath.split("@", 1)[-1]
                
            if sha1_pattern.match(filepath):
                self.hashes[filepath] = filepath
            else:           
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha1(f.read()).hexdigest()
                self.hashes[filepath] = file_hash
                
            if self.rank in [0, -1]:
                w, h = size
                store[self.hashes[filepath]] = {
                    "train_use": True, 
                    "train_caption": caption, 
                    "file_path": filepath,
                    "train_width": w,
                    "train_height": h,
                }

        if self.rank in [0, -1]:
            # save hashes file 
            # json format: {"<image_sha1_hash>": {"train_use": true, "train_caption": ""}}
            with open("dataset.json", "w") as f:
                json.dump(store, f, indent=4)
            
    def hash(self, fp):
        if self.cache_enabled and len(self.hashes) == 0:
            # read hashes from dataset.json
            with open("dataset.json", "r") as f:
                store = json.load(f)
            self.hashes = {v["file_path"]: k for k, v in store.items()}
                    
        if self.allow_duplicates and "@" in filepath:
            filepath = filepath.split("@", 1)[-1]
            
        if self.use_legacy_key:
            return fp
        
        return self.hashes[fp]
        
    def cache_latents(self, vae):
        import h5py
        
        cache_dir = Path(self.cache_dir)
        store = self.buckets
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        with open("cache_index.tmp", "r") as cache:
            cache = json.load(cache)
            cache_keys = set(cache.keys())
            to_cache_images = any(f"{self.hash(img)}.latents" not in cache_keys \
                for entry in store.buckets.keys() for img in store.buckets[entry][:])
            if not to_cache_images:
                rank_zero_print(f"Restored cache from {cache_dir.absolute()}")
                return True
    
        cache_file = cache_dir / f"cache_r{self.rank}.h5"      
        with h5py.File(cache_file, "r+") if cache_file.exists() else h5py.File(cache_file, "w") as cache:
            self.fulfill_cache(cache, vae, store)
            
        return False
    
    def update_cache_index(self, cache_dir, rank):
        if rank not in [0, -1]:
            return

        cache_parts = list(Path(cache_dir).glob("cache_r*.h5"))
        bar = tqdm(desc="Updating index", mininterval=0.25)
        cache_index = {}
        for input_file in cache_parts:
            with h5py.File(input_file, 'r') as fi:
                for key in fi.keys():
                    cache_index[key] = str(input_file)
                    bar.update(1)
        
        # save as json
        with open("cache_index.tmp", "w") as f:
            json.dump(cache_index, f)

    def fulfill_cache(self, cache, vae_encode_func, store):
        progress_bar = tqdm(total=len(self.entries) // self.world_size, \
            desc=f"Caching", disable=self.rank not in [0, -1])
        
        for entry in store.buckets.keys():
            size = store.resolutions[entry]
            imgs = store.buckets[entry][:]
            stride = self.cache_bsz
            start, end = 0, len(imgs)

            if self.world_size > 1:
                # divide the work among the ranks
                per_rank = len(imgs) // self.world_size
                start = self.rank * per_rank
                end = start + per_rank
                
                # handle the case where len(imgs) is not a multiple of self.world_size
                # give the remaining tasks to the last rank
                if self.rank == self.world_size - 1:
                    end = len(imgs)
                    
            for idx in range(start, end, stride):   
                latents_allclose = all([f"{self.hash(img)}.latents" in cache for img in imgs[idx:idx+stride]]) 
                if not latents_allclose:
                    batch = []
                    for img in imgs[idx:idx+stride]:
                        img_data = self.read_img(img)
                        batch.append(self.transformer(img_data, size, center_crop=True))
                        
                    latent = vae_encode_func(torch.stack(batch).cuda())
                    for img, latent in zip(imgs[idx:idx+stride], latent):
                        imghash = self.hash(img)
                        # last check, if exists overwrite
                        if f"{imghash}.latents" in cache:
                            del cache[f"{imghash}.latents"]
                            del cache[f"{imghash}.size"]
                        
                        cache.create_dataset(f"{imghash}.latents", data=latent.detach().squeeze(0).half().cpu().numpy())
                        cache.create_dataset(f"{imghash}.size", data=size)

                progress_bar.update(len(imgs[idx:idx+stride]))
        progress_bar.close()
        
    def build_dict(self, item) -> dict:
        item_id, size, = item["instance"], item["size"],

        if item_id == "":
            return {}
        
        if not hasattr(self, "cache_index") and self.cache_enabled:    
            with open("cache_index.tmp", "r") as f:
                self.cache_index = json.load(f)

        prompt, _ = self.process_tags(self.prompt_cache[item_id])
        if random.random() < self.ucg:
            prompt = ''
            
        example = {f"prompts": prompt}
        if self.cache_enabled:
            import h5py
            cachekey = self.hash(item_id)
            fs_loc = self.cache_index[f"{cachekey}.latents"]
            with h5py.File(fs_loc, "r") as cache:
                latent = torch.asarray(cache[f"{cachekey}.latents"][:])
                original_size = cache[f"{cachekey}.size"][:]
                estimate_size = original_size[1] // 8, original_size[0] // 8,
                if latent.shape != (4, *estimate_size):
                    rank_zero_print(f"Latent shape mismatch for {item_id}! Expected {estimate_size}, got {latent.shape}") 
                    
                w, h = original_size
                _, h1, w1 = latent.shape
                example.update({
                    "images": latent,
                    "original_size_as_tuple": (h, w),
                    "target_size_as_tuple": (h1*8, w1*8),
                    "crop_coords_top_left": (0, 0)
                })
        else:
            base_img = self.read_img(item_id)
            image = self.transformer(base_img, size)
            w, h = size
            example.update({
                "images": image,
                "original_size_as_tuple": (base_img.height, base_img.width),
                "target_size_as_tuple": (h, w),
                "crop_coords_top_left": (0, 0),
            })
            
        return example
    
    def __len__(self):
        return len(self.buckets.res_map) // self.world_size

    def __iter__(self):
        for batch, size in self.buckets.generator():
            for item in batch:
                yield self.build_dict({"size": size, "instance": item})

