# torchrun trainer.py --model_path=/tmp/model --config config/test.yaml

from functools import partial

import pytorch_lightning as pl
import torch
from data.buckets import init_sampler
from data.store import AspectRatioDataset
from hivemind import Float16Compression

from lib.args import parse_args
from lib.model import get_class, load_model
from lib.utils import get_world_size
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import HivemindStrategy

torch.backends.cudnn.benchmark = True

args = parse_args()
config = OmegaConf.load(args.config)

def main(args):
    torch.manual_seed(config.trainer.seed)
    tokenizer, model = load_model(args.model_path, config)
    dataset = AspectRatioDataset(
        tokenizer=tokenizer,
        size=config.trainer.resolution,
        bsz=config.trainer.init_batch_size,
        seed=config.trainer.seed,
        **config.dataset
    )
    
    checkpoint_callback = ModelCheckpoint(**config.checkpoint)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        sampler=init_sampler(
            args, config=config, dataset=dataset, world_size=get_world_size()
        ),
        num_workers=8,
    )
    
    logger = (
        WandbLogger(project=config.monitor.wandb_id)
        if config.monitor.wandb_id != ""
        else None
    )
    
    hivemind = (
        HivemindStrategy(
            scheduler_fn=partial(
                get_class(config.lr_scheduler.name),
                **config.lr_scheduler.params
            ),
            grad_compression=Float16Compression(),
            state_averaging_compression=Float16Compression(),
            **config.hivemind
        )
        if config.trainer.use_hivemind
        else None
    )
    
    trainer = pl.Trainer(
        logger=logger, 
        strategy=hivemind, 
        callbacks=[checkpoint_callback],
        **config.lightning
    )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
