
import itertools
import os
import random
from pathlib import Path

import torch
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

torch.backends.cudnn.benchmark = True


class ImageStore(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        img_path,
        tokenizer,
        size=512,
        center_crop=False,
        pad_tokens=False,
        ucg=0,
        **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.pad_tokens = pad_tokens
        self.ucg = ucg
        self.dataset = img_path
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    def prompt_resolver(self, x):
        img_item = (x, "")
        fp = os.path.splitext(x)[0]

        with open(fp + ".txt") as f:
            content = f.read()
            new_prompt = content
        img_item = (x, new_prompt)

        return img_item
        
    def update_store(self, base):
        print(f"Updating internal dataset: {base}")
        self.entries = [self.prompt_resolver(x) for x in Path(base).iterdir() if x.is_file() and x.suffix != ".txt"]
        self._length = len(self.entries)
        random.shuffle(self.entries)

    def tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    @staticmethod
    def read_img(filepath) -> Image:
        img = Image.open(filepath)

        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def process_tags(tags, min_tags=1, max_tags=32, type_dropout=0.75, keep_important=1.00, keep_jpeg_artifacts=True, sort_tags=False):
        if isinstance(tags, str):
            tags = tags.split(" ")
        final_tags = {}

        tag_dict = {tag: True for tag in tags}
        pure_tag_dict = {tag.split(":", 1)[-1]: tag for tag in tags}
        for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request", "artist_name"]:
            if bad_tag in pure_tag_dict:
                del tag_dict[pure_tag_dict[bad_tag]]

        if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict:
            final_tags["nsfw"] = True

        base_chosen = []
        for tag in tag_dict.keys():
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

        return ", ".join(list(final_tags.keys()))

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

    def update(self, x):
        self.update_store(x)
        for path, prompt in self.entries:
            self.prompt_cache[path] = prompt

    def denormalize(self, img, mean=0.5, std=0.5):
        res = transforms.Normalize((-1*mean/std), (1.0/std))(img.squeeze(0))
        res = torch.clamp(res, 0, 1)
        return res
    
    def collate_fn(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
    
        input_ids = [example["prompt_ids"] for example in examples]
        pixel_values = [example["images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        
        return [input_ids, pixel_values]

    def transformer(self, img, size, center_crop=False):
        x, y = img.size
        short, long = (x, y) if x <= y else (y, x)

        w, h = size
        min_crop, max_crop = (w, h) if w <= h else (h, w)
        ratio_src, ratio_dst = float(long / short), float(max_crop / min_crop)
        
        if (x>y and w<h) or (x<y and w>h):
            # handle i/c mixed input
            img = img.rotate(90, expand=True)
            x, y = img.size
            
        if ratio_src > ratio_dst:
            new_w, new_h = (min_crop, int(min_crop * ratio_src)) if x<y else (int(min_crop * ratio_src), min_crop)
        elif ratio_src < ratio_dst:
            new_w, new_h = (max_crop, int(max_crop / ratio_src)) if x>y else (int(max_crop / ratio_src), max_crop)
        else:
            new_w, new_h = w, h

        image_transforms = transforms.Compose([
            transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((h, w)) if center_crop else transforms.RandomCrop((h, w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        new_img = image_transforms(img)

        if self.debug:
            import uuid

            import torchvision
            print(x, y, "->", new_w, new_h, "->", new_img.shape)
            filename = str(uuid.uuid4())
            torchvision.utils.save_image(self.denormalize(new_img), f"/tmp/{filename}_1.jpg")
            torchvision.utils.save_image(torchvision.transforms.ToTensor()(img), f"/tmp/{filename}_2.jpg")
            print(f"saved: /tmp/{filename}")

        return new_img

    def build_dict(self, item) -> dict:
        item_id, size, = item["instance"], item["size"],
        
        if item_id == "":
            return {}
        
        prompt = self.prompt_cache[item_id]
        image = self.read_img(item_id)
        
        if random.random() < self.ucg:
            prompt = ''
        
        example = {
            f"images": self.transformer(image, size),
            f"prompt_ids": self.tokenize(prompt)
        }
        return example

    def __getitem__(self, index):
        result = []
        for item in index:
            result.append(self.build_dict(item))

        return result

