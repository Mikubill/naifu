
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
    def __init__(self, annotations_file):
        self.base_path = Path(annotations_file)
        # with open("danbooru-tags.json") as file:
        #     self.tag_stat = json.loads(file.read())
        
        entries = []
        for entry in tqdm(self.base_path.iterdir()):
            if entry.is_file():
                entries.append(entry)
        self.process_files(entries)
        
    def process_files(self, files):
        self.entries = []
        progress = tqdm(desc=f"Loading tags")
        for entry in files:
            subentry = []
            with open(entry) as file:
                for line in file:
                    b = json.loads(line)
                    if entry["rating"] != "e":
                        subentry.append(self.parser(entry))
                    progress.update()
            self.entries.extend(subentry)
            
    def parser(self, entry): 
        return entry
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
        return "Tags: " + ", ".join([tag["name"] for tag in self.entries[idx]["tags"]])
    
class MBaseLearningModel(CustomEncoderDiffusionModel):     
    def __init__(self, model_path, config, batch_size):
        super().__init__(model_path, config, batch_size)
        
    def init_model(self):
        super().init_model()
        self.base_model = [UNet2DConditionModel.from_pretrained("/root/dataset/animesfw/unet")]
        self.buckets = list(gen_buckets())
    
    def setup(self, stage):
        # rsync://176.9.41.242:873/danbooru2021/metadata.json.tar.xz
        self.dataset = TagDataset("/root/dataset/metadata")
    
    def train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.dataset,
            num_workers=self.config.dataset.num_workers,
            batch_size=self.batch_size,
            persistent_workers=True,
        )
        return self.train_dataloader
        
    def training_step(self, batch, batch_idx):
        input_ids = batch
        encoder_hidden_states = self.encode_tokens(input_ids).to(self.unet.dtype)
        bucket = random.sample(self.buckets)

        # Sample noise that we'll add to the latents
        noise = torch.randn(bucket)
        bsz = batch.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()

        # Predict the noise residual
        noise_pred = self.unet(noise, timesteps, encoder_hidden_states).sample
        noise_pred_prev = self.base_model[0](noise, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred.float(), noise_pred_prev.float(), reduction="mean")
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss