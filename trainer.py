# python trainer.py --model_path=/tmp/model --config config/test.yaml
import copy
import os

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl

from lib.args import parse_args
from lib.model import StableDiffusionModel

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
from lightning.pytorch.loggers import WandbLogger
from data.store import AspectRatioDataset

def setup_torch(config):
    major, minor = torch.__version__.split('.')[:2]
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
    model = StableDiffusionModel(model_path, config)    
    arb_config = {
        "bsz": config.trainer.batch_size,
        "seed": config.trainer.seed,
        "world_size": farbic.world_size,
        "global_rank": farbic.global_rank,
        **config.arb
    }

    # init Dataset
    dataset = AspectRatioDataset(
        arb_config=arb_config,
        size=config.trainer.resolution,
        seed=config.trainer.seed,
        rank=farbic.global_rank,
        world_size=farbic.world_size,
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

def get_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    items = sorted(os.listdir(checkpoint_dir))
    if not items:
        return None
    return os.path.join(checkpoint_dir, items[-1])

@contextmanager
def ema_scope(sd, enabled=False, context=None):
    if enabled:
        sd.model_ema.store(sd.model.parameters())
        sd.model_ema.copy_to(sd.model)
        if context is not None:
            print(f"{context}: Switched to EMA weights")
    try:
        yield None
    finally:
        if enabled:
            sd.model_ema.restore(sd.model.parameters())
            if context is not None:
                print(f"{context}: Restored training weights")

def train(fabric, model, optimizer, dataloader):
    cfg = model.config.trainer
    grad_accum_steps = cfg.accumulate_grad_batches
    grad_clip_val = cfg.gradient_clip_val
    
    global_step = 0
    current_epoch = 0
    should_stop = False
    
    enabled_sampling = fabric.is_global_zero and model.config.sampling.enabled
    sampling_cfg =  model.config.sampling
    sampling_steps = sampling_cfg.every_n_steps
    sampling_epochs = sampling_cfg.every_n_epochs

    state = {"model": model, "optimizer": optimizer}
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

            if not is_accumulating:
                if grad_clip_val > 0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip_val)
                    
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) 
                global_step += 1

                if cfg.use_ema and fabric.is_global_zero: 
                    model.model_ema(model.model)
                  
            if fabric.is_global_zero:
                prog_bar.update(1)
                prog_bar.set_postfix_str(f"train_loss: {loss:.3f}")
                
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                should_stop = True
                break
            
            if enabled_sampling and sampling_steps > 0 and global_step % sampling_steps == 0:
                with ema_scope(model, enabled=cfg.use_ema, context=None):
                    sampler(fabric.logger, sampling_cfg, model, current_epoch, global_step)
            
        current_epoch += 1
        if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
            should_stop = True
            
        if enabled_sampling and sampling_epochs > 0 and current_epoch % sampling_epochs == 0:
            with ema_scope(model, enabled=cfg.use_ema, context=None):
                sampler(fabric.logger, sampling_cfg, model, current_epoch, global_step)
            
        state.update(global_step=global_step, current_epoch=current_epoch)
        if fabric.is_global_zero and cfg.checkpoint_freq > 0 and current_epoch % cfg.checkpoint_freq == 0:
            fabric.save(os.path.join(cfg.checkpoint_dir, f"nd-epoch-{current_epoch:02d}.ckpt"), state)
            
def sampler(logger, config, model, current_epoch, global_step):
    if not any(config.prompts):
        return
        
    save_dir = Path(config.save_dir) 
    save_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
        
    negative_prompts = list(config.negative_prompts) if OmegaConf.is_list(config.negative_prompts) else config.negative_prompts
    prompts = list(config.prompts) if OmegaConf.is_list(config.prompts) else config.prompts
    prompt_to_gen = copy.deepcopy(prompts)
    images = []
    while len(prompt_to_gen) > 0:
        images.extend(model.sample(prompt_to_gen.pop(), negative_prompts.pop(), generator))

    for j, image in enumerate(images):
        image.save(save_dir / f"nd_sample_e{current_epoch}_s{global_step}_{j}.png")
        
    if config.use_wandb and logger:
        logger.log_image(key="samples", images=images, caption=prompts, step=global_step)

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

def main(args):
    config = OmegaConf.load(args.config)
    config.trainer.resume = args.resume
    setup_torch(config)
    if args.model_path != None:
        config.trainer.model_path = args.model_path 

    logger = WandbLogger(project=config.trainer.wandb_id) if config.trainer.wandb_id != "" else None
    fabric = pl.Fabric(loggers=[logger], **config.lightning)
    fabric.launch()
    fabric.seed_everything(config.trainer.seed)
    
    model, dataset, dataloader = setup_model(config, fabric)
    params_to_optim = [{'params': model.model.parameters()}]
    optimizer = get_class(config.optimizer.name)(params_to_optim, **config.optimizer.params)
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    if config.cache.enabled:
        dataset.setup_cache(model.encode_first_stage, model.conditioner)
        
    fabric.barrier()
    torch.cuda.empty_cache()        
    train(fabric, model, optimizer, dataloader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
