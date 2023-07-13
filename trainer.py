# python trainer.py --model_path=/tmp/model --config config/test.yaml
import os

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl

from lib.args import parse_args
from lib.callbacks import HuggingFaceHubCallback, SampleCallback
from lib.model import StableDiffusionModel

from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from data.store import AspectRatioDataset, ImageStore
from pathlib import Path
from tqdm import tqdm

def setup_torch(config):
    major, minor, _ = torch.__version__.split('.')
    if (int(major) > 1 or (int(major) == 1 and int(minor) >= 12)) and torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{device.major}.{device.minor}")
        precision = 'high' if config.lightning.precision == 32 else 'medium'
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

def setup_model(config, farbic):
    model_path = config.trainer.model_path
    dataset_cls = AspectRatioDataset if config.arb.enabled else ImageStore

    if config.get("lora"):
        if config.lora.get("use_locon"):
            from experiment.locon import LoConDiffusionModel
            model = LoConDiffusionModel(model_path, config)
        else:
            from experiment.lora import LoRADiffusionModel
            model = LoRADiffusionModel(model_path, config)
    else:
        model = StableDiffusionModel(model_path, config)
        
    arb_config = {
        "bsz": config.trainer.batch_size,
        "seed": config.trainer.seed,
        "world_size": farbic.world_size,
        "global_rank": farbic.global_rank,
        **config.arb
    }

    # init Dataset
    dataset = dataset_cls(
        arb_config=arb_config,
        size=config.trainer.resolution,
        seed=config.trainer.seed,
        rank=farbic.global_rank,
        world_size=farbic.world_size,
        init=not config.arb.enabled,
        **config.dataset,
        **config.cache
    )
    # init Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        collate_fn=dataset.collate_fn,
        batch_size=config.trainer.batch_size,
        persistent_workers=True
    )
    return model, dataset, dataloader

# learning rate decay scheduler (cosine with warmup)
# def get_lr(it, warmup_iters=2000, lr_decay_iters=10000, min_lr=6e-5, learning_rate=6e-4):
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_iters:
#         return learning_rate * it / warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > lr_decay_iters:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
#     return min_lr + coeff * (learning_rate - min_lr)

def get_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    items = sorted(os.listdir(checkpoint_dir))
    if not items:
        return None
    return os.path.join(checkpoint_dir, items[-1])

def train(fabric, model, optimizer, dataloader):
    cfg = model.config.trainer
    grad_accum_steps = cfg.accumulate_grad_batches
    grad_clip_val = cfg.gradient_clip_val
    
    global_step = 0
    current_epoch = 0
    should_stop = False

    state = {"model": model, "optim": optimizer}
    if Path(cfg.checkpoint_dir).is_dir() and cfg.get("resume"):
        latest_checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)
        remainder = fabric.load(latest_checkpoint_path, state)
        global_step = remainder.pop("global_step")
        current_epoch = remainder.pop("current_epoch")

    if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
        should_stop = True
        
    prog_bar = None
    if fabric.is_global_zero:
        prog_bar = tqdm(dataloader, total=len(dataloader)-1, desc=f"Epoch {current_epoch}")
        
    while not should_stop:
        if fabric.is_global_zero:
            prog_bar.refresh()
            prog_bar.reset()
            prog_bar.set_description(f"Epoch {current_epoch}")
        
        for batch_idx, batch in enumerate(dataloader):
            is_accumulating = global_step % grad_accum_steps != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model(batch)
                fabric.backward(loss / grad_accum_steps)
            
            # determine and set the learning rate for this iteration
            # lr = get_lr(iter_num) if decay_lr else learning_rate
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr

            if not is_accumulating:
                if grad_clip_val > 0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip_val)
                    
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) 
                global_step += 1

                if cfg.use_ema: 
                    model.model_ema.update()
                  
            if fabric.is_global_zero:
                prog_bar.update(1)
                postfix_str = f"train_loss: {loss:.3f}"
                prog_bar.set_postfix_str(postfix_str)
                
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                should_stop = True
                break
            
        current_epoch += 1
        if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
            should_stop = True
            break
            
        state.update(global_step=global_step, current_epoch=current_epoch)
        if fabric.is_global_zero and cfg.checkpoint_freq > 0 and current_epoch % cfg.checkpoint_freq == 0:
            fabric.save(os.path.join(cfg.checkpoint_dir, f"nd-epoch-{current_epoch:04d}.ckpt"), state)
            

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def main(args):
    config = OmegaConf.load(args.config)
    setup_torch(config)
    if args.model_path != None:
        config.trainer.model_path = args.model_path 

    logger = WandbLogger(project=config.monitor.wandb_id) if config.monitor.wandb_id != "" else None
    fabric = pl.Fabric(loggers=logger, **config.lightning)
    fabric.launch()
    fabric.seed_everything(config.trainer.seed)
    
    model, dataset, dataloader = setup_model(config, fabric)
    params_to_optim = [{'params': model.model.parameters()}]
    optimizer = get_class(config.optimizer.name)(params_to_optim, **config.optimizer.params)
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    if config.cache.enabled:
        dataset.setup_cache(model.encode_first_stage, model.conditioner)
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        
    if args.resume:
        config.trainer.resume = True
        
    fabric.barrier()
    torch.cuda.empty_cache()        
    train(fabric, model, optimizer, dataloader)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
