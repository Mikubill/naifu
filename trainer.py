# python trainer.py --model_path=/tmp/model --config config/test.yaml
import copy
import os, sys

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl
import argparse

from common.utils import *
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(self, fabric: pl.Fabric, config: OmegaConf):
        self.fabric = fabric

        model_cls = get_class(config.target)
        model, dataset, dataloader, optimizer, scheduler = model_cls(fabric, config)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader

    def prepare_logger(self):
        fabric = self.fabric
        if fabric.logger and fabric.logger.__class__.__name__ != "CSVLogger":
            config = OmegaConf.to_container(self.model.config, resolve=True)
            fabric.logger.log_hyperparams(config)
            
    def on_post_training_batch(self, global_step, current_epoch, state):
        config = self.model.config
        fabric = self.fabric

        if fabric.logger:
            self.log_lr_values(global_step, fabric)

        self.perform_sampling(global_step, current_epoch)
        self.save_model(global_step, current_epoch, state)

    def log_lr_values(self, global_step, fabric):
        optimizer_name = self.model.config.optimizer.name
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        ocls = self.optimizer.__class__.__name__

        for i, lr in enumerate(last_lr):
            fabric.log(f"lr-{ocls}-{i}", lr, step=global_step)

        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return

        last_d_lr = [(g["d"] * g["lr"]) for g in self.optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            fabric.log(f"d*lr-{ocls}-{i}", lr, step=global_step)

    def save_model(self, global_step, current_epoch, state, is_last=False):
        config = self.model.config
        cfg = config.trainer
        fabric = self.fabric
        ckpt_st = cfg.checkpoint_steps
        ckpt_fq = cfg.checkpoint_freq
        ckpt_dir = cfg.checkpoint_dir

        is_ckpt_step = ckpt_st > 0 and global_step % ckpt_st == 0
        is_ckpt_epoch = ckpt_fq > 0 and current_epoch % ckpt_fq == 0
        to_save = (is_last and is_ckpt_epoch) or is_ckpt_step

        if not (fabric.is_global_zero and to_save):
            return

        model_path = os.path.join(ckpt_dir, f"nd-checkpoint-e{current_epoch:02d}")
        if is_ckpt_step:
            model_path = os.path.join(ckpt_dir, f"nd-checkpoint-s{global_step:02d}")
            
        self.model.save_checkpoint(model_path)

    def perform_sampling(self, global_step, current_epoch, is_last=False):
        config = self.model.config
        enabled_sampling = self.fabric.is_global_zero and config.sampling.enabled
        if not enabled_sampling:
            return

        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = sampling_epochs > 0 and current_epoch % sampling_epochs == 0

        if (is_last and sample_by_epoch) or sample_by_step:
            self.sampler(
                logger=self.fabric.logger, 
                config=sampling_cfg, 
                model=self.model, 
                current_epoch=current_epoch, 
                global_step=global_step
            )

    @rank_zero_only
    def sampler(self, logger, config, model, current_epoch, global_step):
        if not any(config.prompts):
            return

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        prompts = list(config.prompts)
        prompt_to_gen = copy.deepcopy(prompts)
        images = []
        height, width = config.get("height", 1024), config.get("width", 1024)
        size = (height, width)
        for prompt in zip(prompt_to_gen):
            images.extend(model.sample(prompt, size=size, generator=generator))

        for j, image in enumerate(images):
            image.save(save_dir / f"nd_sample_e{current_epoch}_s{global_step}_{j}.png")

        if config.use_wandb and logger and "CSVLogger" != logger.__class__.__name__:
            logger.log_image(
                key="samples", 
                images=images, 
                caption=prompts, 
                step=global_step
            )

        torch.cuda.empty_cache()
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    def train(self):
        config = self.model.config
        cfg = config.trainer
        fabric = self.fabric
        grad_accum_steps = cfg.accumulate_grad_batches
        grad_clip_val = cfg.gradient_clip_val

        local_step = global_step = 0
        current_epoch = 0
        should_stop = False

        state = {"state_dict": self.model}
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        
        if not cfg.get("save_weights_only", False):
            state.update({"optimizer": self.optimizer})

        if Path(cfg.checkpoint_dir).is_dir() and cfg.get("resume"):
            latest_checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)
            remainder = self.fabric.load(latest_checkpoint_path, state)
            global_step = remainder.pop("global_step")
            current_epoch = remainder.pop("current_epoch")
            rank_zero_print(f"Resuming from checkpoint {latest_checkpoint_path}")

        if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
            should_stop = True

        rank_zero_print(f"Starting training from epoch {current_epoch}")
        self.prepare_logger()
        prog_bar = tqdm(
            self.dataloader,
            total=len(self.dataloader) // grad_accum_steps - 1,
            desc=f"Epoch {current_epoch}",
            disable=not fabric.is_global_zero,
        )

        loss_rec = LossRecorder()
        assert len(self.dataloader) > 0, "Dataloader is empty"
        while not should_stop:
            if fabric.is_global_zero:
                prog_bar.refresh()
                prog_bar.reset()
                prog_bar.total = len(self.dataloader) // grad_accum_steps
                prog_bar.set_description(f"Epoch {current_epoch}")

            for batch_idx, batch in enumerate(self.dataloader):
                local_step += 1
                is_accumulating = local_step % grad_accum_steps != 0

                with fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)

                loss_rec.add(epoch=current_epoch, step=batch_idx, loss=loss.item())
                postfix = (
                    f"train_loss: {loss:.3f}, avg_loss: {loss_rec.moving_average:.3f}"
                )
                prog_bar.set_postfix_str(postfix)

                if is_accumulating:
                    # skip here if we are accumulating
                    continue

                if grad_clip_val > 0:
                    val = grad_clip_val
                    self.fabric.clip_gradients(self.model, self.optimizer, max_norm=val)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    fp_batch = current_epoch + batch_idx / len(self.dataloader)
                    if "transformers" in config.scheduler.name:
                        self.scheduler.step(global_step)
                    else:
                        self.scheduler.step(fp_batch)

                if fabric.logger:
                    fabric.log("train_loss", loss, step=global_step)

                global_step += 1
                prog_bar.update(1)
                prog_bar.set_postfix_str(postfix)
                state.update(global_step=global_step, current_epoch=current_epoch)
                self.on_post_training_batch(global_step, current_epoch, state)

            current_epoch += 1
            if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
                should_stop = True

            state.update(global_step=global_step, current_epoch=current_epoch)
            self.perform_sampling(global_step, current_epoch, is_last=True)
            self.save_model(global_step, current_epoch, state, is_last=True)
            
            
class NonAutocastMixedPrecision(pl.fabric.plugins.precision.amp.MixedPrecision):
    """Mixed precision training without automatic casting inputs and outputs."""

    def convert_input(self, data):
        return data

    def convert_output(self, data):
        return data


class LossRecorder:
    def __init__(self):
        self.loss_list = []
        self.loss_total = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0 or len(self.loss_list) <= step:
            self.loss_list.append(loss)
        else:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        if len(self.loss_list) == 0:
            return 0.0
        return self.loss_total / len(self.loss_list)


def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument("--resume", action="store_true")
    first_args = sys.argv[1]
    if first_args.startswith("--"):
        args = parser.parse_args()
    else:
        args = parser.parse_args(sys.argv[2:])
        args.config = first_args
        
    config = OmegaConf.load(args.config)
    config.trainer.resume = args.resume
    plugins = []
    
    strategy = config.lightning.get("strategy", "auto")
    if "." in strategy:
        strategy = get_class(strategy)

    if os.environ.get("SM_TRAINING", False) or os.environ.get("SM_HOSTS", False):
        strategy, config = setup_smddp(config)

    target_precision = config.lightning.precision
    if target_precision in ["16-mixed", "bf16-mixed"]:
        config.lightning.precision = None
        plugins.append(NonAutocastMixedPrecision(target_precision, "cuda"))

    loggers = pl.fabric.loggers.CSVLogger(".")
    if config.trainer.wandb_id != "":
        loggers = WandbLogger(project=config.trainer.wandb_id)
        
    fabric = pl.Fabric(
        loggers=[loggers], 
        plugins=plugins, 
        strategy=strategy, 
        **config.lightning
    )
    fabric.launch()
    fabric.barrier()
    fabric.seed_everything(config.trainer.seed)
    Trainer(fabric, config).train()


if __name__ == "__main__":
    main()
