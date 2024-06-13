# python trainer.py --config config/test.yaml
import os
import time

import random

import torch.utils

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import wandb
import lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import get_worker_info, DistributedSampler
from lightning.fabric.wrappers import _unwrap_objects
from safetensors.torch import save_file

from common.logging import logger
from common.utils import get_class, parse_args, ProgressBar
from common.model import StableDiffusionModel
from common.dataset import AspectRatioDataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    """
    Trains a new SDXL model.
    """
    args = parse_args()
    config = OmegaConf.load(args.config)

    # ==============================
    # Initialize Distributed Training
    # ==============================
    fabric = pl.Fabric(precision=config.trainer.precision)
    fabric.launch()
    fabric.seed_everything(config.trainer.seed)
    
    if fabric.is_global_zero:
        wandb.init(
            project="sd3",
            entity="nyanko",
            config=OmegaConf.to_container(config, resolve=True),
        )

    # ==============================
    # Initialize Model and Dataset
    # ==============================
    model = StableDiffusionModel(
        config=config,
        device=fabric.device,
    )
    dataset = AspectRatioDataset(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )

    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        random.seed(worker_info.seed)  # type: ignore
        torch.manual_seed(worker_info.seed)
        worker_info.dataset.init_batches()

    dist_sampler = DistributedSampler(
        dataset,
        num_replicas=fabric.world_size,
        rank=fabric.local_rank,
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
    params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.clip_l.parameters(), "lr": lr})

    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.clip_g.parameters(), "lr": lr})
        
    if config.advanced.get("train_text_encoder_3"):
        lr = config.advanced.get("text_encoder_3_lr", config.optimizer.params.lr)
        params_to_optim.append({"params": model.t5xxl.parameters(), "lr": lr})

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(params_to_optim, **optim_param)
    lr_scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    # Boost model for distributed training\
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    logger.info("Setup model for distributed training")

    # we dont need to use the model in the first stage
    max_epoch = config.trainer.max_epochs
    current_epoch = global_step = 0
    
    def log_lr_values(optimizer, config, global_step):
        """
        Log learning rate values for the optimizer.
        """
        optimizer_name = config.optimizer.name
        last_lr = [group.get("lr", 0) for group in optimizer.param_groups]
        ocls = optimizer.__class__.__name__

        for i, lr in enumerate(last_lr):
            fabric.log_dict({f"lr/{ocls}-{i}": lr}, step=global_step)

        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return

        last_d_lr = [(g["d"] * g["lr"]) for g in optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            fabric.log_dict({f"d*lr/{ocls}-{i}": lr}, step=global_step)

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

        _unwrap_objects(model).generate_samples(
            world_size=fabric.world_size,
            rank=fabric.global_rank,
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

        postfix = f"e{current_epoch}_s{global_step}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Saving model checkpoint")
        
        # fir 1node only
        if fabric.is_global_zero:
            model_state_dict = {f"model.{k}": v for k,v in model.model.state_dict().items()}
            vae_state_dict = {f"first_stage_model.{k}": v for k,v in model.first_stage_model.state_dict().items()}
            model_state_dict.update(vae_state_dict)
            save_file(model_state_dict, f"{ckpt_dir}/model_{postfix}.safetensors")
            
            if config.advanced.get("train_text_encoder_1"):
                save_file(model.clip_l.state_dict(), f"{ckpt_dir}/clip_l_{postfix}.safetensors")
            if config.advanced.get("train_text_encoder_2"):
                save_file(model.clip_g.state_dict(), f"{ckpt_dir}/clip_g_{postfix}.safetensors")
            if config.advanced.get("train_text_encoder_3"):
                save_file(model.t5xxl.state_dict(), f"{ckpt_dir}/t5xxl_{postfix}.safetensors")
                
            # handle optimizer and other states
            torch.save(optimizer.state_dict(), f"{ckpt_dir}/optimizer_{postfix}.pt")
        fabric.barrier()

    # ==============================
    # Training Loop
    # ==============================
    perform_sampling(forced=True)
    progress = ProgressBar(
        total=len(dataloader),
        disable=not fabric.is_global_zero,
    )
    for current_epoch in range(max_epoch):
        # here we can use the dataloader directly
        desc = f"Epoch {current_epoch}"
        progress.update(desc, 0)

        for batch_idx, batch in enumerate(dataloader):
            # move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(fabric.device)
            
            # setup local timer    
            local_timer = time.perf_counter()
            optimizer.zero_grad()

            # forward pass + backward pass
            loss = model(batch)    
            fabric.backward(loss)
            grad_norm = fabric.clip_gradients(
                module=model, 
                optimizer=optimizer, 
                max_norm=1.0,
                error_if_nonfinite=False,
            )
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            perform_sampling(is_last=False)
            save_model(is_last=False)
            
            if fabric.is_global_zero:
                log_lr_values(optimizer, config, global_step)
                wandb.log({
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                    "trainer/step_t": time.perf_counter() - local_timer,
                    "trainer/global_step": global_step,
                    "trainer/current_epoch": current_epoch,
                })
                
            local_step = batch_idx
            progress.update(desc, local_step, status=f"train_loss: {loss:.3f}")

        # here is the end of the epoch
        perform_sampling(is_last=True)
        save_model(is_last=True)

    logger.info("Done!")

if __name__ == "__main__":
    main()