
# torchrun trainer.py --model_path=/tmp/model --config test-run.yaml

import torch
import pytorch_lightning as pl

from data.buckets import init_sampler
from data.store import AspectRatioDataset

from lib.args import parse_args
from lib.model import load_model
from lib.utils import get_world_size
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True

args = parse_args()
config = OmegaConf.load(args.config)
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda')
world_size = get_world_size()
weight_dtype = torch.float16 if config.trainer.precision == "fp16" else torch.float32

def main(args):
    torch.manual_seed(config.trainer.seed)
    tokenizer, model = load_model(args.model_path, config)
    dataset = AspectRatioDataset(
        tokenizer=tokenizer,
        size=config.trainer.resolution,
        bsz=args.train_batch_size,
        seed=config.trainer.seed,
        **config.dataset
    )
    sampler = init_sampler(
        args, 
        config=config, 
        dataset=dataset, 
        world_size=world_size
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=dataset.collate_fn, 
        sampler=sampler, 
        num_workers=8
    )
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=8, accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=train_dataloader)

if __name__ == "__main__":
    args = parse_args() 
    main(args)