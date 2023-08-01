import cv2, os
import pandas as pd
import numpy as np
import argparse
import re

from typing import Tuple, List, Dict
from PIL import Image
from tqdm.auto import tqdm

from pathlib import Path
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

# Code from https://github.com/toriato/stable-diffusion-webui-wd14-tagger
# Partily from https://github.com/AdjointOperator/End2End-Tagger/

tag_escape_pattern = re.compile(r"([\\()])")

def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im

def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class WDInterrogator():
    def __init__(
        self,
        name: str,
        model_path="model.onnx",
        tags_path="selected_tags.csv",
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs
        
    def load(self):
        model_path, tags_path = self.download()
        self.model = InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
        self.tags = pd.read_csv(tags_path)
        self.useless_tags = set(
            ['virtual_youtuber'] +
            [tag for tag in self.tags if 'alternate_' in tag] +
            ['genderswap', 'genderswap_(mtf)', 'genderswap_(ftm)', 'ambiguous_gender']
        )
        print(f"Loaded {self.name} model from {model_path}")
    
    def download(self):
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")
        model_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(**self.kwargs, filename=self.tags_path))
        return model_path, tags_path
    
    def postprocess_tags(
        self,
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False,
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0
        
        # remove useless tags
        for t in self.useless_tags:
            if t in tags:
                del tags[t]

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c
            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order,
            )
            # filter tags
            if (c >= threshold and t not in exclude_tags)
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace("_", " ")

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r"\\\1", new_tag)

            if add_confident_as_weight:
                new_tag = f"({new_tag}:{tags[tag]})"

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def interrogate(self, image):
        if not hasattr(self, 'model') or self.model is None:
            self.load()
            
        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][["name"]]
        tags["confidents"] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)
        return ratings, tags


interrogators = {
    "wd14-convnextv2-v2": WDInterrogator(
        "wd14-convnextv2-v2",
        repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        revision="v2.0",
    ),
    "wd14-vit-v2": WDInterrogator(
        "wd14-vit-v2", 
        repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2", 
        revision="v2.0"
    ),
    "wd14-convnext-v2": WDInterrogator(
        "wd14-convnext-v2",
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
        revision="v2.0",
    ),
    "wd14-swinv2-v2": WDInterrogator(
        "wd14-swinv2-v2",
        repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        revision="v2.0",
    ),
    "wd14-convnextv2-v2-git": WDInterrogator(
        "wd14-convnextv2-v2",
        repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    ),
    "wd14-vit-v2-git": WDInterrogator(
        "wd14-vit-v2-git", 
        repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2"
    ),
    "wd14-convnext-v2-git": WDInterrogator(
        "wd14-convnext-v2-git", 
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2"
    ),
    "wd14-swinv2-v2-git": WDInterrogator(
        "wd14-swinv2-v2-git", 
        repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2"
    ),
    "wd14-vit": WDInterrogator(
        "wd14-vit", 
        repo_id="SmilingWolf/wd-v1-4-vit-tagger"
    ),
    "wd14-convnext": WDInterrogator(
        "wd14-convnext", 
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger"
    ),
    "wd-v1-4-moat-tagger-v2": WDInterrogator(
        "wd-v1-4-moat-tagger-v2",
        repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2"
    ),
}

if __name__ == "__main__":
    # give a path to folder with images, use tqdm for progress bar
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, default="/notebooks")
    args.add_argument("--interrogator", type=str, default="wd14-swinv2-v2")
    args.add_argument("--threshold", type=float, default=0.5)
    args.add_argument("--prefix", type=str, default="")
    args = args.parse_args()
    
    # iter args.path
    for path in tqdm(os.listdir(args.path)):
        imgpath = os.path.join(args.path, path)
        name, ext = os.path.splitext(os.path.basename(imgpath))
        if ext not in [".jpg", ".png", ".jpeg", ".webp"]:
            continue
        
        interrogator = interrogators[args.interrogator]
        ratings, tags = interrogator.interrogate(Image.open(imgpath))
        tags = interrogator.postprocess_tags(tags, threshold=args.threshold)

        with open(os.path.join(args.path, f"{name}.txt"), "w") as f:
            if args.prefix and tags:
                text_to_write = args.prefix + ", " + ", ".join(tags.keys())
            elif args.prefix:
                text_to_write = args.prefix
            else:
                text_to_write = ", ".join(tags.keys())
            f.write(text_to_write)