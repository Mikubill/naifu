"""Simple script to finetune a stable-diffusion model"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()
    
    if args.config.startswith("https://"):
        print(f"Downloading config {arg.config}...")
        r = requests.get(args.config, stream=True)
        with open("config.yaml", 'wb') as fd:
            for chunk in r.iter_content(chunk_size=256):
                fd.write(chunk)
        args.config = "config.yaml"

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def DownloadFile(url, filename=None):
  if filename is None:
    local_filename = url.split('/')[-1]
  else:
    local_filename = filename
  r = requests.get(url)
  f = open(local_filename, 'wb')
  for chunk in r.iter_content(chunk_size=512 * 1024): 
    if chunk:
      f.write(chunk)
  f.close()