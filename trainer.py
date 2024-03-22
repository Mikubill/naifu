# python trainer.py --config config/test.yaml
import os
import sys
from typing import Dict, Tuple

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl
import argparse

from common.utils import *
from common.logging import logger
from omegaconf import OmegaConf
from pathlib import Path
from lightning.fabric.connector import _is_using_cli


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
        save_method_avail = hasattr(self.model, "save_checkpoint")

        if use_fabric_save:
            self.fabric.save(model_path + ".ckpt", state)
        elif not save_method_avail:
            logger.info("Model does not have a save_checkpoint method, falling back to fabric.save")
            self.fabric.save(model_path + ".ckpt", state)
        else:
            metadata = {
                "global_step": str(self.global_step),
                "current_epoch": str(self.current_epoch),
            }
            self.model.save_checkpoint(model_path, metadata)
            if not save_weights_only:
                optimizer_state = {"optimizer": self.optimizer, **metadata}
                self.fabric.save(model_path + "_optimizer.pt", optimizer_state)

    def perform_sampling(self, is_last: bool = False):
        """
        Perform image/text sampling.

        Args:
            is_last (bool): Indicates if it is the last sampling.
        """
        config = self.model.config
        enabled_sampling = self.fabric.is_global_zero and config.sampling.enabled \
            and hasattr(self.model, "generate_samples")

        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and self.global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = sampling_epochs > 0 and self.current_epoch % sampling_epochs == 0

        if not enabled_sampling or len(sampling_cfg.prompts) == 0:
            return

        if (is_last and sample_by_epoch) or sample_by_step:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()
            if sampling_cfg.get("save_dir", None):
                os.makedirs(sampling_cfg.save_dir, exist_ok=True)

            self.model.generate_samples(
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

        if Path(cfg.checkpoint_dir).is_dir():
            latest_ckpt = get_latest_checkpoint(cfg.checkpoint_dir)

        if cfg.get("resume") and latest_ckpt:
            remainder = {}
            if cfg.get("use_fabric_save", False):
                remainder = self.fabric.load(latest_ckpt, state)
            else:
                remainder = sd = load_torch_file(
                    ckpt=latest_ckpt, 
                    device=self.model.target_device, 
                    extract=False
                )
                if latest_ckpt.endswith(".safetensors"):
                    remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
                fabric.call("load_checkpoint", sd)

                opt = Path(latest_ckpt).stem + "_optimizer"
                opt_path = Path(latest_ckpt).with_stem(opt)
                if opt_path.is_file():
                    remainder = fabric.load(opt_path, {"optimizer": self.optimizer})
                    logger.info(f"Loaded optimizer state from {opt_path}")

            if remainder:
                self.global_step = int(remainder.pop("global_step", 0))
                self.current_epoch = int(remainder.pop("current_epoch", 0))

            logger.info(f"Resuming from checkpoint {latest_ckpt}")

        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True

        logger.info(f"Starting training from epoch {self.current_epoch}")
        self.prepare_logger()

        loss_rec = LossRecorder()
        progress  = ProgressBar(
            total=len(self.dataloader) // config.trainer.accumulate_grad_batches - 1,
            disable=not fabric.is_global_zero,
        )
        assert len(self.dataloader) > 0, "Dataloader is empty"
        while not should_stop:
            desc = f"Epoch {self.current_epoch}"
            progress.update(desc, 0)

            for batch_idx, batch in enumerate(self.dataloader):
                local_step += 1
                local_acc_step = batch_idx // grad_accum_steps
                is_accumulating = local_step % grad_accum_steps != 0

                fabric_module = getattr(self.model, "model", None)
                if hasattr(self.model, "get_module"):
                    fabric_module = self.model.get_module()

                with fabric.no_backward_sync(fabric_module, enabled=is_accumulating):
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)

                loss_rec.add(epoch=self.current_epoch, step=batch_idx, loss=loss.item())
                status = f"train_loss: {loss:.3f}, avg_loss: {loss_rec.avg:.3f}"
                progress.update(desc, local_acc_step, status=status)

                # skip here if we are accumulating
                if is_accumulating:
                    continue

                if grad_clip_val > 0:
                    self.fabric.clip_gradients(
                        module=self.model, 
                        optimizer=self.optimizer, 
                        max_norm=grad_clip_val
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    is_transformers_sch = "transformers" in config.scheduler.name
                    fp_batch = self.current_epoch + batch_idx / len(self.dataloader)
                    actual_step = self.global_step if is_transformers_sch else fp_batch
                    self.scheduler.step(actual_step)

                if fabric.logger:
                    fabric.log("train_loss", loss, step=self.global_step)

                self.global_step += 1
                state.update(global_step=self.global_step, current_epoch=self.current_epoch)
                self.on_post_training_batch(state)

            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True

            state.update(global_step=self.global_step, current_epoch=self.current_epoch)
            self.perform_sampling(is_last=True)
            self.save_model(state, is_last=True)
            self.eval_model(is_last=True)


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

    strategy = config.lightning.pop("strategy", "auto")
    if "." in strategy:
        _params = config.lightning.pop("strategy_params", {})
        strategy = get_class(strategy)(**_params)

    if os.environ.get("SM_TRAINING", False) or os.environ.get("SM_HOSTS", False):
        strategy, config = setup_smddp(config)

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
    if not _is_using_cli():
        fabric.launch()
        
    fabric.barrier()
    fabric.seed_everything(config.trainer.seed)
    Trainer(fabric, config).train_loop()


if __name__ == "__main__":
    main()
