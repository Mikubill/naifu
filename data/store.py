
import functools
import itertools
import json
import os
import random
from pathlib import Path
import tarfile
import tempfile
import types


import requests
from lib.augment import AugmentTransforms

import torch
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class ImageStore(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        img_path,
        size=512,
        center_crop=False,
        max_length=225,
        ucg=0,
        rank=0,
        augment=None,
        process_tags=True,
        **kwargs
    ):
        self.size = size
        self.fliter_tags = process_tags
        self.center_crop = center_crop
        self.ucg = ucg
        self.max_length = max_length
        self.rank = rank
        self.dataset = img_path
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.yandere_tags = {}
        self.augment = AugmentTransforms(augment)
        print()
        
        # https://huggingface.co/datasets/nyanko7/yandere-images/blob/main/yandere-tags.json
        if Path("yandere-tags.json").is_file():
            with open("yandere-tags.json") as f:
                self.yandere_tags = json.loads(f.read())
            print(f"Read {len(self.yandere_tags)} tags from yandere-tags.json")
            
        self.update_store()

    def prompt_resolver(self, x: str):
        img_item = (x, "")
        fp = os.path.splitext(x)[0]

        with open(fp + ".txt") as f:
            content = f.read()
            new_prompt = content

        return x, new_prompt
    
    def download(self, url, mpath="model"):
        print(f'Downloading: "{url}" to {mpath}\n')
        r = requests.get(url, stream=True)
        file_size = int(r.headers.get("content-length", 0))

        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
            file = tarfile.open(fileobj=r_raw, mode="r|gz")
            file.extractall(path=mpath)
            
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def update_store(self):            
        self.entries = []
        bar = tqdm(total=-1, desc=f"Loading captions", disable=self.rank not in [0, -1])
        
        for entry in self.dataset:
            if entry.startswith("https://"):
                dlpath = os.path.join(tempfile.gettempdir(), f"dataset-{self.dataset.index(entry)}")
                Path(dlpath).mkdir(exist_ok=True)
                self.download(entry, dlpath)
                entry = dlpath
            
            for x in Path(entry).iterdir():
                if not (x.is_file() and x.suffix in [".jpg", ".png", ".webp", ".bmp", ".gif", ".jpeg", ".tiff"]):
                    continue
                img, prompt = self.prompt_resolver(x)
                _, skip = self.process_tags(prompt)
                if skip: continue
                self.entries.append((img, prompt))
                bar.update(1)

        self._length = len(self.entries)
        random.shuffle(self.entries)

    def tokenize(self, prompt):
        '''
        Handle token truncation in collate_fn()
        '''
        return self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
        ).input_ids 

    @staticmethod
    def read_img(filepath) -> Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img

    def process_tags(self, tags, min_tags=24, max_tags=72, type_dropout=0.75, keep_important=1.00, keep_jpeg_artifacts=True, sort_tags=False):
        if not self.fliter_tags:
            return tags, False
        
        if isinstance(tags, str):
            tags = tags.split(" ")
        final_tags = {}

        tag_dict = {tag: True for tag in tags}
        pure_tag_dict = {tag.split(":", 1)[-1]: tag for tag in tags}
        for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request", "artist_name"]:
            if bad_tag in pure_tag_dict:
                del tag_dict[pure_tag_dict[bad_tag]]

        if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict or "nsfw" in tag_dict:
            final_tags["nsfw"] = True

       
        base_chosen = []
        for tag in tag_dict.keys():
            # For yande.re tags.
            if len(self.yandere_tags) > 0:
                if tag in self.yandere_tags and self.yandere_tags[tag]["type"] >= 1 and random.random() < keep_important:
                        base_chosen.append(tag)
            
            # For danbooru tags.
            parts = tag.split(":", 1)
            if parts[0] in ["artist", "copyright", "character"] and random.random() < keep_important:
                base_chosen.append(tag)
            if len(parts[-1]) > 1 and parts[-1][0] in ["1", "2", "3", "4", "5", "6"] and parts[-1][1:] in ["boy", "boys", "girl", "girls"]:
                base_chosen.append(tag)
            if parts[-1] in ["6+girls", "6+boys", "bad_anatomy", "bad_hands"]:
                base_chosen.append(tag)

        tag_count = min(random.randint(min_tags, max_tags), len(tag_dict.keys()))
        base_chosen_set = set(base_chosen)
        chosen_tags = base_chosen + [tag for tag in random.sample(list(tag_dict.keys()), tag_count) if tag not in base_chosen_set]
        if sort_tags:
            chosen_tags = sorted(chosen_tags)

        for tag in chosen_tags:
            tag = tag.replace(",", "").replace("_", " ")
            if random.random() < type_dropout:
                if tag.startswith("artist:"):
                    tag = tag[7:]
                elif tag.startswith("copyright:"):
                    tag = tag[10:]
                elif tag.startswith("character:"):
                    tag = tag[10:]
                elif tag.startswith("general:"):
                    tag = tag[8:]
            if tag.startswith("meta:"):
                tag = tag[5:]
            final_tags[tag] = True

        skip_image = False
        for bad_tag in ["comic", "panels", "everyone", "sample_watermark", "text_focus", "tagme"]:
            if bad_tag in pure_tag_dict:
                skip_image = True
        if not keep_jpeg_artifacts and "jpeg_artifacts" in tag_dict:
            skip_image = True
            
        return ", ".join(list(final_tags.keys())), skip_image
    
    def collate_fn(self, examples):
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [example["images"] for example in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        
        return [input_ids, pixel_values]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.entries[index % self._length]
        instance_image = self.read_img(instance_path)

        example["images"] = self.image_transforms(instance_image)
        example["prompt_ids"] = self.tokenize(instance_prompt)
        return example


class AspectRatioDataset(ImageStore):
    def __init__(self, debug_arb=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug_arb
        self.prompt_cache = {}
        
        for path, prompt in self.entries:
            self.prompt_cache[path] = prompt

    def denormalize(self, img, mean=0.5, std=0.5):
        res = transforms.Normalize((-1*mean/std), (1.0/std))(img.squeeze(0))
        res = torch.clamp(res, 0, 1)
        return res
    
    def collate_fn(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        return super().collate_fn(examples=examples)

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

        if self.debug:
            import uuid

            import torchvision
            print(x, y, "->", new_w, new_h, "->", new_img.shape)
            
            basedir = os.path.join(tempfile.gettempdir(), "nd-arb-debug")
            os.makedirs(basedir, exist_ok=True)
            filename = "arb_" + str(uuid.uuid4())[:8]
            rawp = os.path.join(basedir, f"{filename}_raw.jpg")
            trsp = os.path.join(basedir, f"{filename}_transformed.jpg")
            
            torchvision.utils.save_image(self.denormalize(new_img), rawp)
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(img), trsp)
            print(f"saved: {rawp}")
            print(f"saved: {trsp}")

        return new_img

    def build_dict(self, item, roll) -> dict:
        item_id, size, = item["instance"], item["size"],

        if item_id == "":
            return {}

        prompt, _ = self.process_tags(self.prompt_cache[item_id])
        image = self.augment.transform(self.read_img(item_id), roll)

        if random.random() < self.ucg:
            prompt = ''

        example = {
            f"images": self.transformer(image, size),
            f"prompt_ids": self.tokenize(prompt)
        }
        return example

    def __getitem__(self, index):
        result = []
        rs = random.random()
        for item in index:
            result.append(self.build_dict(item, rs))

        return result
