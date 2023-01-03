
import itertools
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel
from torch.utils.data import Dataset, DataLoader
from experiment.custom_encoder import CustomEncoderDiffusionModel
from lib.utils import get_local_rank, get_world_size
from transformers import BertTokenizerFast, CLIPTextModel, CLIPTokenizer

def gen_buckets(base_res=(512, 512), max_size=512 * 768, dim_range=(256, 1024), divisor=64):
    min_dim, max_dim = dim_range
    buckets = set()

    w = min_dim
    while w * min_dim <= max_size and w <= max_dim:
        h = min_dim
        got_base = False
        while w * (h + divisor) <= max_size and (h + divisor) <= max_dim:
            if w == base_res[0] and h == base_res[1]:
                got_base = True
            h += divisor
        if (w != base_res[0] or h != base_res[1]) and got_base:
            buckets.add(base_res)
        buckets.add((w, h))
        w += divisor

    h = min_dim
    while h / min_dim <= max_size and h <= max_dim:
        w = min_dim
        while h * (w + divisor) <= max_size and (w + divisor) <= max_dim:
            w += divisor
        buckets.add((w, h))
        h += divisor

    return sorted(buckets, key=lambda sz: sz[0] * 4096 - sz[1])

class TagDataset(Dataset):
    def __init__(self, annotations_file, config):
        self.base_path = Path(annotations_file)
        # with open("danbooru-tags.json") as file:
        #     self.tag_stat = json.loads(file.read())
        
        entries = []
        for entry in tqdm(self.base_path.glob('**/*')):
            if entry.is_file():
                entries.append(entry)
        self.config = config
        self.process_files(entries)
        
    def process_files(self, files):
        self.entries = []
        progress = tqdm(desc=f"Loading tags", disable=get_local_rank(self.config) not in [0, -1])
        for entry in files:
            subentry = []
            try:
                if get_local_rank(self.config) in [0, -1]:
                    progress.desc = f"Loading {entry}"
                with open(entry) as file:
                    lines = file.readlines()
            except Exception as e:
                if get_local_rank(self.config) in [0, -1]:
                    print(e)
                continue
                
            for line in lines:
                b = json.loads(line)
                if b["rating"] != "e" and b["rating"] != "q":
                    subentry.append(self.parser(b))
                progress.update()
            self.entries.extend(subentry)
        random.shuffle(self.entries)
            
    def parser(self, entry): 
        return ", ".join([tag["name"] for tag in entry["tags"]])
    #     entry["quality_tag"] = ""
        
    #     # catch-all
    #     if int(entry["score"]) < 0:
    #         entry["quality_tag"] = "low quality"
    #     if int(entry["score"]) < -20:
    #         entry["quality_tag"] = "bad quality"
    #     if int(entry["score"]) < 50:
    #         return entry
        
    #     # by genre
    #     for tag in entry["tags"]:
    #         accept = False
    #         if int(tag["category"]) == 1:
    #             if self.tag_stat[tag["name"]]["counts"] >= 150:
    #                 accept = True
    #                 break
                      
    #             pass

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if random.random() < 0.5:
            return "Tags: " + self.entries[idx]
        else:
            return "Tags: masterpiece, high quality, " + self.entries[idx]
    
class MBaseLearningModel(CustomEncoderDiffusionModel):     
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        
    def init_model(self):
        super().init_model()
        self.base_model = UNet2DConditionModel.from_pretrained("/root/dataset/animesfw/unet")
        self.base_tokenizer = CLIPTokenizer.from_pretrained("/root/dataset/animesfw/tokenizer")
        self.base_encoder = CLIPTextModel.from_pretrained("/root/dataset/animesfw/text_encoder")
        self.base_model.requires_grad_(False)
        self.base_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.base_model.enable_xformers_memory_efficient_attention()
        self.base_model.enable_gradient_checkpointing()
        self.buckets = list(gen_buckets())
    
    def setup(self, stage):
        # rsync://176.9.41.242:873/danbooru2021/metadata.json.tar.xz
        self.dataset = TagDataset("/root/dataset/metadata", self.config)
    
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["state_dict"] = {k: v for k, v in checkpoint if k.startswith("unet.")}
    
    def train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            num_workers=self.config.dataset.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
        )
        return self.train_dataloader
    
    def collate_fn(self, input_ids):
        token_ids_1 = self._tokenizer(input_ids, padding="max_length", max_length=self._tokenizer.model_max_length, truncation=True, return_tensors="pt")
        token_ids_2 = self.base_tokenizer(input_ids, padding=True, truncation=True, return_tensors="pt").input_ids
        return token_ids_1, token_ids_2
        
    def training_step(self, batch, batch_idx):
        token_ids_1, token_ids_2 = batch
        hidden_states_down = self.text_encoder(token_ids_1.input_ids, attention_mask=token_ids_1.attention_mask, output_hidden_states=True).last_hidden_state.to(self.unet.dtype)
        hidden_states_up = self.base_encoder(token_ids_2, output_hidden_states=True).last_hidden_state.to(self.unet.dtype)
        
        bsz = hidden_states_down.shape[0]
        bucket = [[512, 512]] # random.sample(self.buckets, 1)
        noise = torch.randn(bsz, 4, bucket[0][0] // 8, bucket[0][1] // 8, device=self.device, dtype=self.unet.dtype)

        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=self.device)
        timesteps = torch.randint(0, 2, (bsz,), dtype=torch.int64, device=self.device)
        noise_pred = self.unet(noise, timesteps, hidden_states_down).sample
        noise_pred_prev = self.base_model(noise, timesteps, hidden_states_up).sample

        loss = F.mse_loss(noise_pred.float(), noise_pred_prev.float(), reduction="mean")
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss