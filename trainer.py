# torchrun trainer.py --model_path=/tmp/model --config config/test.yaml

# from lib.experiments import T5CLIPDiffusionModel
import pytorch_lightning as pl
import torch
from data.buckets import AspectRatioSampler
from data.store import AspectRatioDataset
from lib.args import parse_args
from lib.callbacks import HuggingFaceHubCallback
from lib.model import load_model
from lib.utils import get_world_size
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

args = parse_args()
config = OmegaConf.load(args.config)

def main(args):
    torch.manual_seed(config.trainer.seed)
    
    strategy = 'ddp'
    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)
        
    tokenizer, model = load_model(args.model_path, config)
    # model = T5CLIPDiffusionModel(args.model_path, config)
    # tokenizer = model.tokenizer
    
    callbacks = [ModelCheckpoint(**config.checkpoint)]
    if config.monitor.huggingface_repo != "":
        hf_logger = HuggingFaceHubCallback(
            config.monitor.huggingface_repo, 
            use_auth_token=config.monitor.hf_auth_token
        )
        callbacks.append(hf_logger)
    
    dataset = AspectRatioDataset(
        tokenizer=tokenizer,
        size=config.trainer.resolution,
        bsz=config.trainer.init_batch_size,
        seed=config.trainer.seed,
        rank=args.local_rank,
        **config.dataset
    )
    
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
        callbacks=callbacks,
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
