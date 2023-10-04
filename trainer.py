# python trainer.py --model_path=/tmp/model --config config/test.yaml
import copy
import os
import h5py
import re

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl

from lib.args import parse_args
from lib.model import StableDiffusionModel
from lib.precision import HalfPrecisionPlugin
from lib.lora import LoConBaseModel

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from lightning.pytorch.loggers import WandbLogger
from data.store import AspectRatioDataset
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only

def setup_torch(config):
    major, minor = torch.__version__.split('.')[:2]
    if (int(major) > 1 or (int(major) == 1 and int(minor) >= 12)) and torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{device.major}.{device.minor}")
        precision = 'high' if config.lightning.precision in ["32", "32-true"] else 'medium'
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

def setup_model(config, farbic):
    model_path = config.trainer.model_path
    model = StableDiffusionModel(model_path, config, farbic.device) 
    
    arb_config = {
        "bsz": config.trainer.batch_size,
        "seed": config.trainer.seed,
        "world_size": farbic.world_size,
        "global_rank": farbic.global_rank,
    }
    if config.get("arb", None) is None:
        # calculate arb from resolution
        base = config.trainer.resolution
        arb_config.update({
            "base_res": (base, base),
            "max_size": (base, base),
            "divisible": 64,
            "max_ar_error": 1,
            "min_dim": int(base // 2),
            "dim_limit": int(base * 2),
            "debug": False
        })
    else:
        arb_config.update(**config.arb) 

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

def train(fabric: pl.Fabric, model, optimizer, scheduler, dataset, dataloader):
    cfg = model.config.trainer
    grad_accum_steps = cfg.accumulate_grad_batches
    grad_clip_val = cfg.gradient_clip_val
    
    local_step = global_step = 0
    current_epoch = 0
    should_stop = False
    
    enabled_sampling = fabric.is_global_zero and model.config.sampling.enabled
    sampling_cfg =  model.config.sampling
    sampling_steps = sampling_cfg.every_n_steps
    sampling_epochs = sampling_cfg.every_n_epochs
    
    state = {"state_dict": model}
        
    if not cfg.get("save_weights_only", False):
        state.update({"optimizer": optimizer})
        
    if enabled_sampling:
        size = (sampling_cfg.height, sampling_cfg.width)
        assert size[0] % 64 == 0 and size[1] % 64 == 0, "Sample image size must be a multiple of 64"
        
    if Path(cfg.checkpoint_dir).is_dir() and cfg.get("resume"):
        latest_checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)
        remainder = fabric.load(latest_checkpoint_path, state)
        global_step = remainder.pop("global_step")
        current_epoch = remainder.pop("current_epoch")

    if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
        should_stop = True
        
    ema_ctx = nullcontext
    if cfg.use_ema:
        ema_ctx = model.model_ema.average_parameters
        model.model_ema.to(fabric.device)
        
    lora_ctx = nullcontext
    if cfg.get("lora") and cfg.lora.enabled:
        lora_ctx = model.lora.apply_weights
        model.lora.to(fabric.device)
        
    prog_bar = None
    if fabric.is_global_zero:
        if fabric.logger:
            fabric.logger.experiment.config.update(OmegaConf.to_container(model.config, resolve=True))
        prog_bar = tqdm(dataloader, total=len(dataloader) // grad_accum_steps - 1, desc=f"Epoch {current_epoch}")

    while not should_stop:
        if model.config.cache.get("precompute_embeds", False):
            conditioner = model.get_conditioner()
            fabric.to_device(conditioner)
            dataset.precompute_embeds(conditioner, prog_bar)
            # free conditioner from memory
            conditioner.cpu()
            
        if fabric.is_global_zero:
            prog_bar.refresh()
            prog_bar.reset()
            prog_bar.total = len(dataloader) // grad_accum_steps - 1
            prog_bar.set_description(f"Epoch {current_epoch}")
            
        assert len(dataloader) > 0, "Dataloader is empty, please check your dataset"
        for batch_idx, batch in enumerate(dataloader):
            local_step += 1  
            is_accumulating = local_step % grad_accum_steps != 0
            
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model(batch)
                fabric.backward(loss / grad_accum_steps)
                
            if is_accumulating:
                # skip here if we are accumulating
                continue

            global_step += 1
            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                should_stop = True
                break
            
            if enabled_sampling and sampling_steps > 0 and global_step % sampling_steps == 0:
                with ema_ctx(), lora_ctx():
                    sampler(fabric.logger, sampling_cfg, model, current_epoch, global_step)

            if fabric.is_global_zero and cfg.checkpoint_steps > 0 and global_step % cfg.checkpoint_steps == 0:
                with ema_ctx(), lora_ctx():
                    fabric.save(os.path.join(cfg.checkpoint_dir, f"nd-step-{global_step}.ckpt"), state)
                    
            if grad_clip_val > 0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip_val)
                    
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) 
            last_lr = [group.get("lr", 0) for group in optimizer.param_groups]
                
            # use float as epoch
            if scheduler is not None:
                if "transformers" in model.config.scheduler.name:
                    scheduler.step(global_step)
                else:
                    scheduler.step(current_epoch + batch_idx / len(dataloader))
                    
            if cfg.wandb_id != "":
                fabric.log("train_loss", loss, step=global_step)
                for i, lr in enumerate(last_lr):
                    fabric.log(f"lr-{optimizer.__class__.__name__}-{i}", lr, step=global_step)

            if cfg.use_ema and fabric.is_global_zero: 
                model.model_ema.update()

            if fabric.is_global_zero:
                prog_bar.update(1)
                prog_bar.set_postfix_str(f"train_loss: {loss:.3f}")    
            
        current_epoch += 1
        if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
            should_stop = True
        
        state.update(global_step=global_step, current_epoch=current_epoch)
        with ema_ctx(), lora_ctx():
            if enabled_sampling and sampling_epochs > 0 and current_epoch % sampling_epochs == 0:
                sampler(fabric.logger, sampling_cfg, model, current_epoch, global_step)    
            
            if fabric.is_global_zero and cfg.checkpoint_freq > 0 and current_epoch % cfg.checkpoint_freq == 0:
                fabric.save(os.path.join(cfg.checkpoint_dir, f"nd-epoch-{current_epoch:02d}.ckpt"), state)
            
@rank_zero_only
def sampler(logger, config, model, current_epoch, global_step):
    if not any(config.prompts):
        return
    
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
    save_dir = Path(config.save_dir) 
    save_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
        
    negative_prompts = list(config.negative_prompts) if OmegaConf.is_list(config.negative_prompts) else config.negative_prompts
    prompts = list(config.prompts) if OmegaConf.is_list(config.prompts) else config.prompts
    prompt_to_gen = copy.deepcopy(prompts)
    images = []
    size = (config.height, config.width)
    for prompt, negative_prompt in zip(prompt_to_gen, negative_prompts):
        images.extend(model.sample(prompt, negative_prompt, generator, size=size))

    for j, image in enumerate(images):
        image.save(save_dir / f"nd_sample_e{current_epoch}_s{global_step}_{j}.png")
        
    if config.get("callback", None):
        get_class(config.callback)(images=images, caption=prompts, step=global_step, logger=logger)
        
    if config.use_wandb and logger:
        logger.log_image(key="samples", images=images, caption=prompts, step=global_step)
    
    torch.cuda.empty_cache()
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)

def cast_precision(tensor, precision):
    if precision == "bf16":
        tensor.to(torch.bfloat16)
    elif precision == "fp16":
        tensor.to(torch.float16)
    else:
        tensor.to(precision)
    return tensor

def create_vds_for_group(source_group, target_group, bar):
    for key in source_group.keys():
        if key in target_group:
            if key.endswith(".latents"): bar.update(1)
            continue
        item = source_group.get(key)
        layout = h5py.VirtualLayout(shape=item.shape, dtype=item.dtype)
        layout[:] = h5py.VirtualSource(item)
        target_group.create_virtual_dataset(key, layout)
        if key.endswith(".latents"): bar.update(1)

@rank_zero_only
def update_cache_index(cache_dir):
    os.remove("cache_index.tmp") if os.path.exists("cache_index.tmp") else None
    try:
        cache_parts = list(Path(cache_dir).glob("cache_r*.h5"))
        with h5py.File("cache_index.tmp", 'a', libver='latest', driver='core') as fo:  # using 'latest' for VDS support
            bar = tqdm(desc="Updating index")
            for input_file in cache_parts:
                with h5py.File(input_file, 'r') as fi:
                    create_vds_for_group(fi, fo, bar)
    except Exception as e:
        print(f"Warn: unable to write cache_index.tmp - remove if exists: {e}")
        exit()
                
def get_lr_for_name(name, lr_conf):
    for item in lr_conf:
        regex_pattern, lr = list(item.items())[0]
        if re.match(regex_pattern, name):
            return lr
    return None

def main(args):
    config = OmegaConf.load(args.config)
    config.trainer.resume = args.resume
    
    setup_torch(config)
    if args.model_path != None:
        config.trainer.model_path = args.model_path 
        
    plugins = []
    strategy = config.lightning.get("strategy", "auto")
    if config.get("lora") and config.lora.enabled:
        from lightning.fabric.strategies import DDPStrategy
        strategy = DDPStrategy(static_graph=True) if torch.cuda.device_count() > 1 else "auto"
    
    model_precision = config.trainer.get("model_precision", torch.float32)
    target_precision = config.lightning.precision
    if target_precision in ["16-true", "bf16-true"]:
        plugins.append(HalfPrecisionPlugin(target_precision))
        model_precision = torch.float16 if target_precision == "16-true" else torch.bfloat16
        del config.lightning.precision

    logger = WandbLogger(project=config.trainer.wandb_id) if config.trainer.wandb_id != "" else None
    fabric = pl.Fabric(loggers=[logger], plugins=plugins, strategy=strategy, **config.lightning)
    fabric.launch()
    fabric.seed_everything(config.trainer.seed)
    
    model, dataset, dataloader = setup_model(config, fabric)
    lr_conf = config.optimizer.get("layer_lr", None)
    if lr_conf:
        lr_groups = {}  # Dictionary to store parameters grouped by their LR
        params_without_lr = []
        
        for name, param in model.model.diffusion_model.named_parameters():
            layer_lr = get_lr_for_name(name, lr_conf)
            if layer_lr:
                if layer_lr not in lr_groups:
                    lr_groups[layer_lr] = []
                lr_groups[layer_lr].append(param)
            else:
                params_without_lr.append(param)

        params_to_optim = []
        for lr, params in lr_groups.items():
            params_to_optim.append({'params': params, 'lr': lr})
        if params_without_lr:
            params_to_optim.append({'params': params_without_lr})
    else:
        params_to_optim = [{'params': model.model.parameters()}]   

    if config.get("lora") and config.lora.enabled:
        lora = LoConBaseModel(model.model, config.lora)
        for param in model.parameters():
            param.requires_grad = False
        
        lora.requires_grad_(True)
        lora.inject()
        model.lora = lora
        params_to_optim = [{'params': lora.parameters(), 'lr': config.lora.unet_lr}]

    optimizer = get_class(config.optimizer.name)(params_to_optim, **config.optimizer.params)
    if fabric.is_global_zero and os.name != 'nt':
        print(f"\n{ModelSummary(model, max_depth=1)}")

    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)
    
    fabric.to_device(model)
    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)
    dataloader = fabric.setup_dataloaders(dataloader)

    if config.cache.enabled:
        update_cache_index(config.cache.cache_dir)
        fabric.barrier()
        
        model.first_stage_model.to(torch.float32)
        allclose = dataset.setup_cache(model.encode_first_stage, model.get_conditioner())
        fabric.barrier()
        if not allclose:
            update_cache_index(config.cache.cache_dir)
            
    if model_precision:
        cast_precision(model, model_precision)
        model.first_stage_model.to(torch.float32)

    fabric.barrier()
    torch.cuda.empty_cache()        
    train(fabric, model, optimizer, scheduler, dataset, dataloader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
