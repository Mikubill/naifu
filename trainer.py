# python trainer.py --model_path=/tmp/model --config config/test.yaml
import os
import sys
from typing import Dict, Tuple

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl
import argparse

from common.utils import *
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(self, fabric: pl.Fabric, config: OmegaConf):
        """
        Initialize the trainer with the given fabric and configuration.

        Args:
            fabric (pl.Fabric): The PyTorch Lightning Fabric instance.
            config (OmegaConf): The configuration object.
        """
        self.fabric = fabric

        model_cls = get_class(config.target)
        model, dataset, dataloader, optimizer, scheduler = model_cls(fabric, config)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader
        self.global_step, self.current_epoch = 0, 0

    def prepare_logger(self):
        """Prepare the logger and log hyperparameters if the logger is not CSVLogger."""
        fabric = self.fabric
        if fabric.logger and fabric.logger.__class__.__name__ != "CSVLogger":
            config = OmegaConf.to_container(self.model.config, resolve=True)
            fabric.logger.log_hyperparams(config)

    def on_post_training_batch(self, state: Dict):
        """
        Perform actions after each training batch.

        Args:
            state (Dict): The state dictionary.
        """
        if self.fabric.logger:
            self.log_lr_values()

        self.perform_sampling()
        self.save_model(state)
        self.eval_model()

    def log_lr_values(self):
        """
        Log learning rate values for the optimizer.

        Args:
            self.global_step (int): The current global step.
        """
        optimizer_name = self.model.config.optimizer.name
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        ocls = self.optimizer.__class__.__name__

        for i, lr in enumerate(last_lr):
            self.fabric.log(f"lr-{ocls}-{i}", lr, step=self.global_step)

        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return

        last_d_lr = [(g["d"] * g["lr"]) for g in self.optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            self.fabric.log(f"d*lr-{ocls}-{i}", lr, step=self.global_step)

    def eval_model(self, is_last: bool = False):
        """
        Save the model checkpoint.

        Args:
            self.global_step (int): The current global step.
            self.current_epoch (int): The current epoch.
            state (Dict): The state dictionary.
            is_last (bool): Indicates if it is the last checkpoint.
        """
        config = self.model.config
        cfg = config.trainer
        eval_st = cfg.get("eval_steps", -1)
        eval_fq = cfg.get("eval_epochs", -1)

        is_eval_step = eval_st > 0 and self.global_step % eval_st == 0
        is_eval_epoch = eval_fq > 0 and self.current_epoch % eval_fq == 0
        should_eval = (is_last and is_eval_epoch) or is_eval_step
        if not should_eval:
            return
        
        self.fabric.call(
            "eval_model",
            logger=self.fabric.logger,
            current_epoch=self.current_epoch,
            global_step=self.global_step,
        )

    def save_model(self, state: Dict, is_last: bool = False):
        """
        Save the model checkpoint.

        Args:
            self.global_step (int): The current global step.
            self.current_epoch (int): The current epoch.
            state (Dict): The state dictionary.
            is_last (bool): Indicates if it is the last checkpoint.
        """
        config = self.model.config
        cfg = config.trainer
        ckpt_st = cfg.checkpoint_steps
        ckpt_fq = cfg.checkpoint_freq
        ckpt_dir = cfg.checkpoint_dir

        is_ckpt_step = ckpt_st > 0 and self.global_step % ckpt_st == 0
        is_ckpt_epoch = ckpt_fq > 0 and self.current_epoch % ckpt_fq == 0
        should_save = (is_last and is_ckpt_epoch) or is_ckpt_step
        if not should_save:
            return

        postfix = f"e{self.current_epoch}_s{self.global_step}"
        model_path = os.path.join(ckpt_dir, f"checkpoint-{postfix}")
        use_fabric_save = cfg.get("use_fabric_save", False)
        save_weights_only = cfg.get("save_weights_only", False)
        
        if use_fabric_save:
            self.fabric.save(model_path + ".ckpt", state)
        else:
            self.fabric.call("save_checkpoint", model_path)
            if not save_weights_only:
                self.fabric.save(model_path + "_optimizer.pt", {"optimizer": self.optimizer})

    def perform_sampling(self, is_last: bool = False):
        """
        Perform image sampling.

        Args:
            is_last (bool): Indicates if it is the last sampling.
        """
        config = self.model.config
        enabled_sampling = self.fabric.is_global_zero and config.sampling.enabled

        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and self.global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = (
            sampling_epochs > 0 and self.current_epoch % sampling_epochs == 0
        )

        if not enabled_sampling or len(sampling_cfg.prompts) == 0:
            return

        if (is_last and sample_by_epoch) or sample_by_step:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()
            if sampling_cfg.get("save_dir", None):
                os.makedirs(sampling_cfg.save_dir, exist_ok=True)

            self.fabric.call(
                "generate_samples",
                logger=self.fabric.logger,
                current_epoch=self.current_epoch,
                global_step=self.global_step,
            )

            torch.cuda.empty_cache()
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)

    def train_loop(self):
        """Run the training loop."""
        config = self.model.config
        cfg = config.trainer
        fabric: pl.Fabric = self.fabric
        grad_accum_steps = cfg.accumulate_grad_batches
        grad_clip_val = cfg.gradient_clip_val

        local_step = 0
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        state = {"state_dict": self.model, "optimizer": self.optimizer}

        if Path(cfg.checkpoint_dir).is_dir() and cfg.get("resume"):
            latest_checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)

            if not cfg.save_weights_only:  # use normal fabric save
                remainder = self.fabric.load(latest_checkpoint_path, state)
                self.global_step = remainder.pop("self.global_step")
                self.current_epoch = remainder.pop("self.current_epoch")
            else:
                fabric.call("load_checkpoint", latest_checkpoint_path)
                ckpt_path = Path(latest_checkpoint_path)
                parent, ckpt_stem = ckpt_path.parent, ckpt_path.stem
                opt_path = parent / (ckpt_stem + "_optimizer.pt")
                if opt_path.is_file():
                    fabric.load(opt_path, {"optimizer": self.optimizer})

            rank_zero_print(f"Resuming from checkpoint {latest_checkpoint_path}")

        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True

        rank_zero_print(f"Starting training from epoch {self.current_epoch}")
        self.prepare_logger()
        prog_bar = tqdm(
            self.dataloader,
            total=len(self.dataloader) // grad_accum_steps - 1,
            desc=f"Epoch {self.current_epoch}",
            disable=not fabric.is_global_zero,
        )

        loss_rec = LossRecorder()
        assert len(self.dataloader) > 0, "Dataloader is empty"
        while not should_stop:
            if fabric.is_global_zero:
                prog_bar.refresh()
                prog_bar.reset()
                prog_bar.total = len(self.dataloader) // grad_accum_steps
                prog_bar.set_description(f"Epoch {self.current_epoch}")

            for batch_idx, batch in enumerate(self.dataloader):
                local_step += 1
                is_accumulating = local_step % grad_accum_steps != 0

                with fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)

                loss_rec.add(epoch=self.current_epoch, step=batch_idx, loss=loss.item())
                postfix = f"train_loss: {loss:.3f}, avg_loss: {loss_rec.avg:.3f}"
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
                    fp_batch = self.current_epoch + batch_idx / len(self.dataloader)
                    if "transformers" in config.scheduler.name:
                        self.scheduler.step(self.global_step)
                    else:
                        self.scheduler.step(fp_batch)

                if fabric.logger:
                    fabric.log("train_loss", loss, step=self.global_step)

                self.global_step += 1
                prog_bar.update(1)
                prog_bar.set_postfix_str(postfix)
                state.update(
                    global_step=self.global_step, 
                    current_epoch=self.current_epoch
                )
                self.on_post_training_batch(state)
                
            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True

            state.update(global_step=self.global_step, current_epoch=self.current_epoch)
            self.perform_sampling(is_last=True)
            self.save_model(state, is_last=True)
            self.eval_model(is_last=True)


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
    def avg(self) -> float:
        # return the average loss of the last epoch
        if len(self.loss_list) == 0:
            return 0.0
        return self.loss_total / len(self.loss_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
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

    if "strategy" in config.lightning:
        del config.lightning.strategy

    if os.environ.get("SM_TRAINING", False) or os.environ.get("SM_HOSTS", False):
        strategy, config = setup_smddp(config)

    target_precision = config.lightning.precision
    if target_precision in ["16-mixed", "bf16-mixed"]:
        config.lightning.precision = None
        plugins.append(NonAutocastMixedPrecision(target_precision, "cuda"))

    loggers = pl.fabric.loggers.CSVLogger(".")
    if config.trainer.wandb_id != "":
        from lightning.pytorch.loggers import WandbLogger
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
    Trainer(fabric, config).train_loop()


if __name__ == "__main__":
    main()
