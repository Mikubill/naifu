# torchrun trainer.py --model_path=/tmp/model --config config/test.yaml

from functools import partial

import pytorch_lightning as pl
import torch
from data.buckets import AspectRatioSampler
from data.store import AspectRatioDataset

from lib.args import parse_args
from lib.model import load_model
from lib.utils import get_world_size
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

args = parse_args()
config = OmegaConf.load(args.config)

def main(args):
    torch.manual_seed(config.trainer.seed)
    
    strategy = None
    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)
        
    tokenizer, model = load_model(args.model_path, config)
    dataset = AspectRatioDataset(
        tokenizer=tokenizer,
        size=config.trainer.resolution,
        bsz=config.trainer.init_batch_size,
        seed=config.trainer.seed,
        rank=args.local_rank,
        **config.dataset
    )
    
    checkpoint_callback = ModelCheckpoint(**config.checkpoint)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        sampler=AspectRatioSampler(config, args.local_rank, dataset, get_world_size()),
        num_workers=config.dataset.num_workers,
        persistent_workers=True,
    )
    
    logger = (
        WandbLogger(project=config.monitor.wandb_id)
        if config.monitor.wandb_id != ""
        else None
    )
    
    trainer = pl.Trainer(
        logger=logger, 
        strategy=strategy, 
        callbacks=[checkpoint_callback],
        **config.lightning
    )
    
    trainer.fit(
        model=model, 
        ckpt_path=args.resume if args.resume else None,
        train_dataloaders=train_dataloader, 
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
