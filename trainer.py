# python trainer.py --model_path=/tmp/model --config config/test.yaml

import torch
import pytorch_lightning as pl

from lib.args import parse_args
from lib.callbacks import HuggingFaceHubCallback
from lib.model import load_model

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

args = parse_args()
config = OmegaConf.load(args.config)

def main(args):
    torch.manual_seed(config.trainer.seed)
    
    strategy = None
    if not config.trainer.gradient_checkpointing and config.lightning.accelerator in ["gpu", "cpu"]:
        strategy = "ddp_find_unused_parameters_false"
         
    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)
        
    model = load_model(args.model_path, config)
    
    # for stable diffusion 2.0, use it with --model_path stabilityai/stable-diffusion-2
    # from experiment.models import MultiEncoderDiffusionModel
    # model = MultiEncoderDiffusionModel(args.model_path, config, config.trainer.init_batch_size)
    
    callbacks = [ ModelCheckpoint(**config.checkpoint)]
    if config.monitor.huggingface_repo != "":
        hf_logger = HuggingFaceHubCallback(
            config.monitor.huggingface_repo, 
            use_auth_token=config.monitor.hf_auth_token
        )
        callbacks.append(hf_logger)
    
    logger = None
    if config.monitor.wandb_id != "":
        logger = WandbLogger(project=config.monitor.wandb_id)
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    trainer = pl.Trainer(
        logger=logger, 
        strategy=strategy, 
        callbacks=callbacks,
        **config.lightning
    )
    
    if trainer.auto_scale_batch_size or trainer.auto_lr_find:
        trainer.tune(model=model, scale_batch_size_kwargs={"steps_per_trial": 5})
    
    trainer.fit(
        model=model,
        ckpt_path=args.resume if args.resume else None
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
