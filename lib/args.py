import argparse
import os

import requests


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
        default="train.yaml"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from previous saved checkpoint.",
    )

    args = parser.parse_args()
    
    if args.config != None and args.config.startswith("https://"):
        print(f"Downloading config {args.config}...")
        r = requests.get(args.config, stream=True)
        with open("config.yaml", 'wb') as fd:
            for chunk in r.iter_content(chunk_size=256):
                fd.write(chunk)
        args.config = "config.yaml"

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
