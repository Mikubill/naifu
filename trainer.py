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
    
    if config.get("lora"):
        if config.lora.get("use_locon"):
            from experiment.locon import LoConDiffusionModel
            model = LoConDiffusionModel(args.model_path, config, config.trainer.batch_size)
        else:
            from experiment.lora import LoRADiffusionModel
            model = LoRADiffusionModel(args.model_path, config, config.trainer.batch_size)
        strategy = config.lightning.strategy = None
    else:
        model = load_model(args.model_path, config)

    # for ddp-optimize only
    # from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
    # strategy = pl.strategies.DDPStrategy(
    #     find_unused_parameters=False,
    #     gradient_as_bucket_view=True,
    #     ddp_comm_state=post_localSGD.PostLocalSGDState(
    #         process_group=None,
    #         subgroup=None,
    #         start_localSGD_iter=8,
    #     ),
    #     ddp_comm_hook=post_localSGD.post_localSGD_hook,
    #     model_averaging_period=4,
    # )
    
    # for experiment only
    # from experiment.attn_realign import AttnRealignModel
    # model = AttnRealignModel(args.model_path, config, config.trainer.init_batch_size)
    # from experiment.kwlenc import MixinModel
    # model = MixinModel(args.model_path, config, config.trainer.init_batch_size)
    
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
        
    if config.get("custom_embeddings") != None and config.custom_embeddings.enabled:
        from experiment.textual_inversion import CustomEmbeddingsCallback
        callbacks.append(CustomEmbeddingsCallback(config.custom_embeddings))
        if not config.custom_embeddings.train_all and not config.custom_embeddings.concepts.trainable:
            if strategy == 'ddp':
                strategy = 'ddp_find_unused_parameters_false'
        if config.custom_embeddings.freeze_unet:
            if strategy == 'ddp_find_unused_parameters_false':
                strategy = 'ddp'
        
    if config.get("sampling") != None and config.sampling.enabled:
        callbacks.append(SampleCallback(config.sampling, logger))
        
    if config.lightning.get("strategy") is None:
        config.lightning.strategy = strategy

    if not config.get("custom_embeddings") or not config.custom_embeddings.freeze_unet:
        callbacks.append(ModelCheckpoint(**config.checkpoint))
        enable_checkpointing = True
    else:
        enable_checkpointing = False

    if config.lightning.get("enable_checkpointing") == None:
        config.lightning.enable_checkpointing = enable_checkpointing
    
    trainer = pl.Trainer(
        logger=logger, 
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

