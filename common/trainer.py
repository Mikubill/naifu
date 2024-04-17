import os
import time

import torch
import lightning as pl

from common.utils import *
from common.logging import logger
from omegaconf import OmegaConf
from pathlib import Path

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
        self.global_step = int(config.get("global_step", 0))
        self.current_epoch = int(config.get("current_epoch", 0))

    def prepare_logger(self):
        """Prepare the logger and log hyperparameters if the logger is not CSVLogger."""
        fabric = self.fabric
        if fabric.logger and fabric.logger.__class__.__name__ != "CSVLogger":
            config = OmegaConf.to_container(self.model.config, resolve=True)
            fabric.logger.log_hyperparams(config)

    def on_post_training_batch(self, is_last=False):
        """
        Perform actions after each training batch.
        """
        if self.fabric.logger and not is_last:
            self.log_lr_values()

        self.perform_sampling(is_last=is_last)
        self.save_model(is_last=is_last)
        self.eval_model(is_last=is_last)

    def log_lr_values(self):
        """
        Log learning rate values for the optimizer.
        """
        optimizer_name = self.model.config.optimizer.name
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        ocls = self.optimizer.__class__.__name__

        for i, lr in enumerate(last_lr):
            self.fabric.log(f"lr/{ocls}-{i}", lr, step=self.global_step)

        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return

        last_d_lr = [(g["d"] * g["lr"]) for g in self.optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            self.fabric.log(f"d*lr/{ocls}-{i}", lr, step=self.global_step)

    def eval_model(self, is_last: bool = False):
        """
        Save the model checkpoint.
        """
        config = self.model.config
        cfg = config.trainer
        eval_st = cfg.get("eval_steps", -1)
        eval_fq = cfg.get("eval_epochs", -1)

        is_eval_step = eval_st > 0 and self.global_step % eval_st == 0
        is_eval_epoch = eval_fq > 0 and self.current_epoch % eval_fq == 0
        
        should_eval = (is_last and is_eval_epoch) or is_eval_step
        has_eval_method = hasattr(self.model, "eval_model")
        
        if not should_eval or not has_eval_method:
            return

        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.eval()
            
        self.model.eval_model(
            logger=self.fabric.logger,
            current_epoch=self.current_epoch,
            global_step=self.global_step,
        )
        torch.cuda.empty_cache()
        
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def save_model(self, is_last: bool = False):
        """
        Save the model checkpoint.
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
        
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.eval()

        postfix = f"e{self.current_epoch}_s{self.global_step}"
        model_path = os.path.join(ckpt_dir, f"checkpoint-{postfix}")
        save_weights_only = cfg.get("save_weights_only", False)
        
        logger.info("Saving model checkpoint")
        metadata = {
            "global_step": str(self.global_step),
            "current_epoch": str(self.current_epoch),
        }
        
        # # save state if you want to use fabric.save
        # state = dict(
        #     mode=self.model,
        #     global_step=self.global_step, 
        #     current_epoch=self.current_epoch,
        #     optimizer=self.optimizer,
        # )
        # self.fabric.save(model_path + "_state.pt", state)
        
        self.model.save_checkpoint(model_path, metadata)
        if not save_weights_only:
            optimizer_state = {"optimizer": self.optimizer, **metadata}
            self.fabric.save(model_path + "_optimizer.pt", optimizer_state)
            
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def perform_sampling(self, is_last: bool = False):
        """
        Perform image/text sampling.
        """
        config = self.model.config
        enabled_sampling = config.sampling.enabled and hasattr(self.model, "generate_samples")

        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and self.global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = sampling_epochs > 0 and self.current_epoch % sampling_epochs == 0

        if not enabled_sampling or len(sampling_cfg.prompts) == 0:
            return
        
        if sampling_cfg.get("save_dir", None):
            os.makedirs(sampling_cfg.save_dir, exist_ok=True)

        if (is_last and sample_by_epoch) or sample_by_step:
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.eval()
                
            torch.cuda.empty_cache()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()

            self.model.generate_samples(
                logger=self.fabric.logger,
                current_epoch=self.current_epoch,
                global_step=self.global_step,
            )
            torch.cuda.empty_cache()
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()
            
    def train_loop(self):
        """
        Run the training loop.
        """
        config = self.model.config
        fabric = self.fabric
        cfg = config.trainer
        
        grad_accum_steps = cfg.accumulate_grad_batches
        grad_clip_val = cfg.gradient_clip_val

        local_step = 0
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        latest_ckpt = get_latest_checkpoint(cfg.checkpoint_dir)
        
        # # load state if you used fabric.save
        # state = dict(
        #     mode=self.model,
        #     global_step=self.global_step, 
        #     current_epoch=self.current_epoch,
        #     optimizer=self.optimizer,
        # )
        # self.fabric.load(model_path + "_state.pt", state)

        if cfg.get("resume") and latest_ckpt:
            opt_name = Path(latest_ckpt).stem + "_optimizer"
            opt_path = Path(latest_ckpt).with_stem(opt_name).with_suffix(".pt")
            if opt_path.is_file():
                remainder = fabric.load(opt_path, {"optimizer": self.optimizer})
                logger.info(f"Loaded optimizer state from {opt_path}")
                self.global_step = int(remainder.pop("global_step", self.global_step))
                self.current_epoch = int(remainder.pop("current_epoch", self.current_epoch))

        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True

        logger.info(f"Starting training from epoch {self.current_epoch}")
        self.prepare_logger()

        loss_rec = LossRecorder()
        progress  = ProgressBar(
            total=len(self.dataloader) // config.trainer.accumulate_grad_batches,
            disable=not fabric.is_global_zero,
        )
        assert len(self.dataloader) > 0, "Dataloader is empty"
        while not should_stop:
            desc = f"Epoch {self.current_epoch}"
            progress.update(desc, 0)
            torch.cuda.empty_cache()
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()

            for batch_idx, batch in enumerate(self.dataloader):                
                local_step += 1
                
                if local_step == 2:
                    self.perform_sampling(is_last=True)
                    
                local_acc_step = batch_idx // grad_accum_steps + 1
                local_timer = time.perf_counter()
                
                is_accumulating = local_step % grad_accum_steps != 0
                fabric_module = getattr(self.model, "model", None)
                if hasattr(self.model, "get_module"):
                    fabric_module = self.model.get_module()
                    
                with fabric.no_backward_sync(fabric_module, enabled=is_accumulating):
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)

                loss = loss.detach().item()
                loss_rec.add(epoch=self.current_epoch, step=batch_idx, loss=loss)
                metrics = {
                    "train/loss": loss,
                    "trainer/step_t": time.perf_counter() - local_timer,
                }
                stat_str = f"train_loss: {loss:.3f}, avg_loss: {loss_rec.avg:.3f}"
                progress.update(desc, local_acc_step, status=stat_str)
                    
                # skip here if we are accumulating
                if is_accumulating:
                    continue

                if grad_clip_val > 0:
                    grad_norm = self.fabric.clip_gradients(
                        module=fabric_module, 
                        optimizer=self.optimizer, 
                        max_norm=grad_clip_val
                    )
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm

                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    is_transformers_sch = "transformers" in config.scheduler.name
                    fp_batch = self.current_epoch + batch_idx / len(self.dataloader)
                    actual_step = self.global_step if is_transformers_sch else fp_batch
                    self.scheduler.step(actual_step)

                if fabric.logger:
                    fabric.log_dict(metrics=metrics, step=self.global_step)

                self.global_step += 1
                self.on_post_training_batch()

            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True

            self.on_post_training_batch(is_last=True)