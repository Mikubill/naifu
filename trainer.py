# python trainer.py --config config/test.yaml
import os
import sys

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl

from common.utils import get_class, parse_args, create_scaled_precision_plugin
from common.trainer import Trainer
from omegaconf import OmegaConf
from lightning.fabric.connector import _is_using_cli

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.trainer.resume = args.resume
    plugins = []

    strategy = config.lightning.pop("strategy", "auto")
    if "." in strategy:
        _params = config.lightning.pop("strategy_params", {})
        strategy = get_class(strategy)(**_params)

    loggers = pl.fabric.loggers.CSVLogger(".")
    if config.trainer.wandb_id != "":
        from lightning.pytorch.loggers import WandbLogger
        loggers = WandbLogger(project=config.trainer.wandb_id)
        
    if config.lightning.precision == "16-true-scaled":
        config.lightning.precision = None
        plugins.append(create_scaled_precision_plugin())

    fabric = pl.Fabric(
        loggers=[loggers], 
        plugins=plugins, 
        strategy=strategy, 
        **config.lightning
    )
    if not _is_using_cli():
        fabric.launch()
        
    fabric.barrier()
    fabric.seed_everything(config.trainer.seed)
    Trainer(fabric, config).train_loop()

if __name__ == "__main__":
    main()