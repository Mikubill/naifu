# python trainer.py --model_path=/tmp/model --config config/test.yaml

import torch
import pytorch_lightning as pl

from lib.args import parse_args
from lib.callbacks import HuggingFaceHubCallback, SampleCallback
from lib.model import load_model

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

args = parse_args()
config = OmegaConf.load(args.config)

def main(args):
    torch.manual_seed(config.trainer.seed)
    if args.model_path == None:
        args.model_path = config.trainer.model_path
    
    strategy = None
    tune = config.lightning.auto_scale_batch_size or config.lightning.auto_lr_find
    if config.lightning.accelerator in ["gpu", "cpu"] and not tune:
        strategy = "ddp_find_unused_parameters_false"
         
    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)
        
        
    model = load_model(args.model_path, config)
    
    # for experiment only
    # from experiment.models import MultiEncoderDiffusionModel
    # model = MultiEncoderDiffusionModel(args.model_path, config, config.trainer.init_batch_size)
    # from experiment.lora import LoRADiffusionModel
    # model = LoRADiffusionModel(args.model_path, config, config.trainer.init_batch_size)
    # strategy = "ddp"
    
    callbacks = []
    if config.monitor.huggingface_repo != "":
        hf_logger = HuggingFaceHubCallback(
            repo_name=config.monitor.huggingface_repo, 
            use_auth_token=config.monitor.hf_auth_token,
            **config.monitor
        )
        callbacks.append(hf_logger)
    
    logger = None
    if config.monitor.wandb_id != "":
        logger = WandbLogger(project=config.monitor.wandb_id)
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
    sp =  config.get("sampling")
    if sp != None and sp.enabled:
        callbacks.append(SampleCallback(sp, logger))
    
    callbacks.append(ModelCheckpoint(**config.checkpoint))
    trainer = pl.Trainer(
        logger=logger, 
        strategy=strategy, 
        callbacks=callbacks,
        **config.lightning
    )
    
    if config.trainer.precision == "fp16" and config.lightning.precision == 16:
        from pytorch_lightning.plugins import PrecisionPlugin
        precision_plugin = PrecisionPlugin()
        precision_plugin.precision = config.lightning.precision
        trainer.strategy.precision_plugin = precision_plugin

    if trainer.auto_scale_batch_size or trainer.auto_lr_find:
        trainer.tune(model=model, scale_batch_size_kwargs={"steps_per_trial": 5})
    
    trainer.fit(
        model=model,
        ckpt_path=args.resume if args.resume else None
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
