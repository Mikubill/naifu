import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms

torch.backends.cudnn.benchmark = True


class AspectRatioBucket:
    '''
    Code from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py

    BucketManager impls NovelAI Aspect Ratio Bucketing, which may greatly improve the quality of outputs according to Novelai's blog (https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac)
    Requires a pickle with mapping of dataset IDs to resolutions called resolutions.pkl to use this.
    '''

    def __init__(self,
        id_size_map,
        max_size=(768, 512),
        divisible=64,
        step_size=8,
        min_dim=256,
        base_res=(512, 512),
        bsz=1,
        world_size=1,
        global_rank=0,
        max_ar_error=4,
        seed=42,
        dim_limit=1024,
        debug=True,
    ):
        if global_rank == -1:
            global_rank = 0
            
        self.res_map = id_size_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = self.get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)

        # separate prng for sharding use for increased thread resilience
        self.epoch_prng = self.get_prng(epoch_seed)
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    @staticmethod
    def get_prng(seed):
        return np.random.RandomState(seed)
    
    def __len__(self):
        return len(self.res_map) // self.bsz

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(
            res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(
            list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            try:
                print(f"skipped images: {skipped}")
                print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
                for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                    print(
                        f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
                print(f"assign_buckets: {timer:.5f}s")
            except Exception as e:
                pass

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = sorted(list(self.res_map.keys()))
        index_len = len(index)
        
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        # if self.debug:
            # print("perm", self.global_rank, index[0:16])
        
        index = index[self.global_rank::self.world_size]
        self.batch_total = len(index) // self.bsz
        assert (len(index) % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = [post_id for post_id in self.buckets[bucket_id] if post_id in index]
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(
                f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [
                    len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id])
                                for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(
                    bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert (found_batch or len(self.left_over)
                    >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " +
                  ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()


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

        self.instance_entries = []
        self.class_entries = []

        def prompt_resolver(x):
            img_item = (x, "")
            fp = os.path.splitext(x)[0]

            with open(fp + ".txt") as f:
                content = f.read()
                new_prompt = content
            img_item = (x, new_prompt)

            return img_item

        self.instance_entries = [prompt_resolver(x) for x in Path(self.dataset).iterdir() if x.is_file() and x.suffix != ".txt"]
        self.num_instance_images = len(self.instance_entries)
        self.num_class_images = len(self.class_entries)
        self._length = max(self.num_class_images, self.num_instance_images)
        
        random.shuffle(self.instance_entries)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

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
        instance_path, instance_prompt = self.instance_entries[index % self.num_instance_images]
        instance_image = self.read_img(instance_path)
            
        example["images"] = self.image_transforms(instance_image)
        example["prompt_ids"] = self.tokenize(instance_prompt)
        return example


class AspectRatioDataset(ImageStore):
    def __init__(self, debug_arb=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug_arb
        self.prompt_cache = {}

        # cache prompts for reading
        for path, prompt in self.instance_entries:
            self.prompt_cache[path] = prompt

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
        
        if (x>y and w<h) or (x<y and w>h) and self.with_prior_preservation:
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
            import uuid, torchvision
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


class AspectRatioSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        buckets: AspectRatioBucket, 
        num_replicas: int = 1,
    ):
        super().__init__(None)
        self.buckets = buckets
        self.num_replicas = num_replicas
    
    def __iter__(self):
        for batch, size in self.buckets.generator():
            yield [{"size": size, "instance": item} for item in batch]
            
    def __len__(self):
        return len(self.buckets) // self.num_replicas
