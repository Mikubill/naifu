# python trainer.py --config config/test.yaml
import os
import time
import torch
import torch.distributed
import torch.utils
import wandb.util

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import wandb
from pathlib import Path
from omegaconf import OmegaConf

from common.logging import logger
from common.utils import get_class, parse_args, ProgressBar
from common.model import FluxModel
from common.dataset import AspectRatioDataset, worker_init_fn
from torch.utils.data import get_worker_info, DistributedSampler

import deepspeed
from deepspeed import zero

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
    
def main():
    """
    Trains a new SDXL model.
    """
    args = parse_args()
    config = OmegaConf.load(args.config)
    
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda", rank)
    # deepspeed.init_distributed()

    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    
    batch_size = config.trainer.batch_size
    global_batch_size = batch_size * world_size
    ds_config = {
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},  
        "zero_optimization": {"stage": 2},
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
        "train_micro_batch_size_per_gpu": batch_size,
        "train_batch_size": global_batch_size,
    } 
    
    config.ds_config = ds_config
    if global_rank in [0, -1]:
        wandb.init(
            project="flux",
            config=OmegaConf.to_container(config, resolve=True),
        )
            
    # ==============================
    # Initialize Model and Dataset
    # ==============================
    model = FluxModel(config=config, device=device) 
    logger.info("Initializing dataset")
    
    dataset = AspectRatioDataset(
        batch_size=config.trainer.batch_size,
        rank=global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dist_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=None,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )
    # ==============================
    # Initialize Optimizer
    # ==============================

    optim_param = config.optimizer.params
    # modify here to use text encoder

    params_to_optim = [{'params': model.model.parameters()}]        
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    lr_scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    # Boost model for distributed training\
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )
    logger.info("Setup model for distributed training")

    # we dont need to use the model in the first stage
    max_epoch = config.trainer.max_epochs
    ckpt_dir = config.trainer.checkpoint_dir
    model_engine.load_checkpoint(f"{ckpt_dir}")
    current_epoch = model_engine.global_steps // len(dataloader)
    global_step = model_engine.global_steps
    
    def perform_sampling(forced=False, is_last=False):
        """
        Perform image/text sampling.
        """
        sampling_cfg = config.sampling
        enabled_sampling = sampling_cfg.enabled or forced
        sample_by_step = sampling_cfg.every_n_steps > 0 and global_step % sampling_cfg.every_n_steps == 0
        sample_by_epoch = sampling_cfg.every_n_epochs > 0 and current_epoch % sampling_cfg.every_n_epochs == 0

        if not enabled_sampling or not ((is_last and sample_by_epoch) or sample_by_step or forced):
            return

        if sampling_cfg.get("save_dir", None):
            os.makedirs(sampling_cfg.save_dir, exist_ok=True)

        torch.cuda.empty_cache()
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        with zero.GatheredParameters(model.model.parameters()): # todo: other modules
            model.generate_samples(
                world_size=world_size,
                rank=global_rank,
                current_epoch=current_epoch,
                global_step=global_step,
            )
        torch.cuda.empty_cache()
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    def save_model(is_last: bool = False):
        """
        Save the model checkpoint.
        """
        model_cfg = config.trainer
        ckpt_st = model_cfg.checkpoint_steps
        ckpt_fq = model_cfg.checkpoint_freq
        ckpt_dir = model_cfg.checkpoint_dir

        is_ckpt_step = ckpt_st > 0 and global_step % ckpt_st == 0
        is_ckpt_epoch = ckpt_fq > 0 and current_epoch % ckpt_fq == 0
        should_save = (is_last and is_ckpt_epoch) or is_ckpt_step
        if not should_save:
            return

        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Saving model checkpoint")

        model_engine.save_checkpoint(f"{ckpt_dir}")
        torch.distributed.barrier()

    # ==============================
    # Training Loop
    # ==============================
    
    progress = ProgressBar(
        total=len(dataloader),
        disable=not global_rank in [0, -1],
    )
    for current_epoch in range(max_epoch):
        # here we can use the dataloader directly
        desc = f"Epoch {current_epoch}"
        progress.update(desc, 0)

        for batch_idx, batch in enumerate(dataloader):
            # move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
                    
            if batch_idx == 0 and current_epoch == 0:
                perform_sampling(forced=True)

            local_timer = time.perf_counter()
            optimizer.zero_grad()

            # forward pass + backward pass
            loss = model(batch)    
            model_engine.backward(loss)
            model_engine.step()

            global_step += 1
            perform_sampling(is_last=False)
            save_model(is_last=False)
            
            if global_rank in [0, -1]:
                wandb.log({
                    "trainer/loss": loss.item(),
                    "trainer/grad_norm": model_engine.get_global_grad_norm(),
                    "trainer/step_t": time.perf_counter() - local_timer,
                    "trainer/global_step": global_step,
                    "trainer/current_epoch": current_epoch,
                })

            local_step = batch_idx
            progress.update(desc, local_step)

        # here is the end of the epoch
        perform_sampling(is_last=True)
        save_model(is_last=True)

    logger.info("Done!")

if __name__ == "__main__":
    main()
