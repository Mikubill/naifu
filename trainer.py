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
from lib.lora import LoConBaseModel
from lib.utils import rank_zero_print

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from lightning.pytorch.loggers import WandbLogger
from data.dataset import AspectRatioDataset, worker_init_fn
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
from lightning.fabric.wrappers import _unwrap_objects
from lightning.fabric.plugins.precision.amp import MixedPrecision
from safetensors.torch import save_model
from contextlib import ExitStack, contextmanager

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

def setup_model(config, fabric):
    model_path = config.trainer.model_path
    model = StableDiffusionModel(model_path, config, fabric.device) 
    
    dataset = AspectRatioDataset(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        base_len=config.trainer.resolution,
        **config.dataset
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=None,
        batch_size=None,
        persistent_workers=False,
        num_workers=config.dataset.get("num_workers", 4),
        worker_init_fn=worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )    
    return model, dataset, dataloader

def get_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    items = sorted(os.listdir(checkpoint_dir))
    if not items:
        return None
    return os.path.join(checkpoint_dir, items[-1])

class Trainer():
    def __init__(self, fabric, model, optimizer, scheduler, dataset, dataloader):
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader
        self.context = []
        
    def prepare_ctx(self):
        cfg = self.model.config.trainer
        ema_ctx = nullcontext
        if cfg.use_ema:
            ema_ctx = self.model.model_ema.average_parameters
            self.model.model_ema.to(self.fabric.device)
            
        lora_ctx = nullcontext
        if cfg.get("lora") and cfg.lora.enabled:
            lora_ctx = self.model.lora.apply_weights
            self.model.lora.to(self.fabric.device)
        
        self.context = [ema_ctx, lora_ctx]
            
    @contextmanager
    def trainer_ctx(self):
        with ExitStack() as stack:
            yield [stack.enter_context(cls()) for cls in self.context]
            
    def prepare_logger(self):
        fabric = self.fabric
        if fabric.logger:
            config = OmegaConf.to_container(self.model.config, resolve=True)
            logger_config = fabric.logger.experiment.config
            if isinstance(logger_config, dict):
                logger_config.update(config)
            
    def on_post_training_batch(self, global_step, current_epoch, state):
        config = self.model.config
        cfg = config.trainer
        fabric = self.fabric
        
        if fabric.logger:
            self.log_lr_values(global_step, fabric)
        
        if cfg.use_ema: 
            self.model.model_ema.update()
            
        self.perform_sampling(global_step, current_epoch)
        self.save_model(global_step, current_epoch, state)
            
    def log_lr_values(self, global_step, fabric):
        optimizer_name = self.model.config.optimizer.name        
        is_adaptive = optimizer_name.startswith("DAdapt") or optimizer_name.startswith("prodigyopt")
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        
        if is_adaptive: 
            last_d_lr = [(group["d"] * group["lr"]) for group in self.optimizer.param_groups]
            
        for i, lr in enumerate(last_lr):
            fabric.log(f"lr-{self.optimizer.__class__.__name__}-{i}", lr, step=global_step)
                    
        if is_adaptive:
            for i, lr in enumerate(last_d_lr):
                fabric.log(f"d*lr-{self.optimizer.__class__.__name__}-{i}", lr, step=global_step)

    def save_model(self, global_step, current_epoch, state, is_last=False):
        config = self.model.config
        cfg = config.trainer
        fabric = self.fabric
        is_checkpoint_step = cfg.checkpoint_steps > 0 and global_step % cfg.checkpoint_steps == 0
        to_save = is_checkpoint_step or is_last

        if not (fabric.is_global_zero and to_save):
            return
        
        with self.trainer_ctx():
            model_path = os.path.join(cfg.checkpoint_dir, f"nd-checkpoint-e{current_epoch:02d}")
            if is_checkpoint_step:
                model_path = os.path.join(cfg.checkpoint_dir, f"nd-checkpoint-s{global_step:02d}")
                
            if cfg.get("save_format", "safetensors"):
                string_cfg = OmegaConf.to_yaml(config)
                model_path += ".safetensors"
                save_model(_unwrap_objects(self.model), model_path, metadata={"trainer_cfg": string_cfg})
            else:
                model_path += ".ckpt"
                fabric.save(model_path, state)
                
            rank_zero_print(f"Saved model to {model_path}")
                    
    def perform_sampling(self, global_step, current_epoch):
        config = self.model.config
        enabled_sampling = self.fabric.is_global_zero and config.sampling.enabled
        sampling_cfg =  config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        if enabled_sampling and sampling_steps > 0 and global_step % sampling_steps == 0:
            size = (sampling_cfg.height, sampling_cfg.width)
            assert size[0] % 64 == 0 and size[1] % 64 == 0, "Sample image size must be a multiple of 64"
            with self.trainer_ctx():
                self.sampler(self.fabric.logger, sampling_cfg, self.model, current_epoch, global_step)
      
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

        if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
            should_stop = True
        
        rank_zero_print(f"Starting training from epoch {current_epoch}")
        self.prepare_ctx()
        self.prepare_logger()
        prog_bar = tqdm(
            self.dataloader, 
            total=len(self.dataloader) // grad_accum_steps - 1, 
            desc=f"Epoch {current_epoch}",
            disable=not fabric.is_global_zero,
        )
        
        loss_rec = LossRecorder()
        while not should_stop:
            if fabric.is_global_zero:
                prog_bar.refresh()
                prog_bar.reset()
                prog_bar.total = len(self.dataloader) // grad_accum_steps
                prog_bar.set_description(f"Epoch {current_epoch}")
            
            import time
            assert len(self.dataloader) > 0, "Dataloader is empty, please check your dataset"
            for batch_idx, batch in enumerate(self.dataloader):
                local_step += 1  
                is_accumulating = local_step % grad_accum_steps != 0
                
                with fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)
                    
                if fabric.is_global_zero:  
                    loss_rec.add(epoch=current_epoch, step=batch_idx, loss=loss.item())
                    prog_bar.set_postfix_str(f"train_loss: {loss:.3f}, avg_loss: {loss_rec.moving_average:.3f}") 
        
                if is_accumulating:
                    # skip here if we are accumulating
                    continue
                
                if grad_clip_val > 0:
                    self.fabric.clip_gradients(self.model, self.optimizer, max_norm=grad_clip_val)
                        
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True) 
                
                if self.scheduler is not None:
                    fp_batch = current_epoch + batch_idx / len(self.dataloader)
                    self.scheduler.step(global_step if "transformers" in config.scheduler.name else fp_batch)
                
                if fabric.logger:
                    fabric.log("train_loss", loss, step=global_step)
                
                global_step += 1
                prog_bar.update(1)
                prog_bar.set_postfix_str(f"train_loss: {loss:.3f}, avg_loss: {loss_rec.moving_average:.3f}")  
                state.update(global_step=global_step, current_epoch=current_epoch)
                self.on_post_training_batch(global_step, current_epoch, state)
                
            current_epoch += 1
            if cfg.max_epochs > 0 and current_epoch >= cfg.max_epochs:
                should_stop = True
            
            state.update(global_step=global_step, current_epoch=current_epoch)
            self.perform_sampling(global_step, current_epoch)
            self.save_model(global_step, current_epoch, state, is_last=True)
            

class LossRecorder:
    def __init__(self):
        self.loss_list = []
        self.loss_total = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
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

def setup_smddp():
    from lightning.pytorch.plugins.environments import LightningEnvironment
    from lightning.fabric.strategies import DDPStrategy

    env = LightningEnvironment()
    env.world_size = lambda: int(os.environ["WORLD_SIZE"])
    env.global_rank = lambda: int(os.environ["RANK"])
    strategy = DDPStrategy(
        cluster_environment=env,  
        accelerator="gpu",
        static_graph=True,
    )
    
    world_size = int(os.environ["WORLD_SIZE"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_nodes = int(world_size/num_gpus)
    init_params = {
        "devices": num_gpus, 
        "num_nodes": num_nodes,
    }
    return strategy, init_params
    
def setup_optimizer(config, model):
    lr_conf = config.optimizer.get("layer_lr", None)
    if lr_conf:
        lr_groups = [{"params": [], "lr": list(it.items())[1]} for it in lr_conf]
        params_without_lr = []
        
        for name, param in model.model.diffusion_model.named_parameters():
            for idx, item in enumerate(lr_conf):
                regex_pattern, lr = list(item.items())[0]
                if re.match(regex_pattern, name):
                    lr_groups[idx]["params"].append(param)
                    continue
            params_without_lr.append(param)
            
        params_without_lr = [{"params": params_without_lr}]
        params_to_optim = [*lr_groups, *params_without_lr]
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
        
    if config.get("advanced"):
        if config.advanced.get("train_text_encoder_1"):
            lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
            params_to_optim.append({'params': model.conditioner.embedders[0].parameters(), 'lr': lr})
        if config.advanced.get("train_text_encoder_2"):
            lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
            params_to_optim.append({'params': model.conditioner.embedders[1].parameters(), 'lr': lr})
            
    for embedder in model.conditioner.embedders[2:]:
        if embedder.is_trainable:
            params_to_optim.append({'params': embedder.parameters()}) 

    optimizer = get_class(config.optimizer.name)(params_to_optim, **config.optimizer.params)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)
    return optimizer, scheduler

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
        
    if config.trainer.get("deepspeed", False):
        assert torch.cuda.device_count() > 1, "DeepSpeed requires multiple GPUs"
        from lightning.fabric.strategies import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(
            config = {"gradient_clipping": 1.0},
            contiguous_memory_optimization=True,
        )
            
    if os.environ.get('SM_TRAINING', False):
        strategy, init_params = setup_smddp()
        config.lightning.update(init_params)
        del config.lightning.accelerator
        
        config.trainer.checkpoint_dir = os.path.join("/opt/ml/checkpoints", config.trainer.checkpoint_dir)
        config.sampling.save_dir = os.path.join(os.environ.get('SM_OUTPUT_DIR'), config.sampling.save_dir)
    
    model_precision = config.trainer.get("model_precision", torch.float32)
    target_precision = config.lightning.precision
    
    if target_precision == "16-true":
        model_precision = torch.float16
        config.lightning.precision = None
        
        # create and patch scaler
        scaler = torch.cuda.amp.GradScaler()
        org_unscale_grads = scaler._unscale_grads_
        def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
            return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        scaler._unscale_grads_ = _unscale_grads_replacer

        plugins.append(MixedPrecision("16-mixed", "cuda", scaler))
    elif target_precision == "bf16-true":
        model_precision = torch.bfloat16
        config.lightning.precision = None
        plugins.append(MixedPrecision("bf16-mixed", "cuda"))

    loggers = WandbLogger(project=config.trainer.wandb_id) if config.trainer.wandb_id != "" else None
    fabric = pl.Fabric(loggers=[loggers], plugins=plugins, strategy=strategy, **config.lightning)
    fabric.launch()
    fabric.seed_everything(config.trainer.seed)
    
    model, dataset, dataloader = setup_model(config, fabric)
    optimizer, scheduler = setup_optimizer(config, model)
    if fabric.is_global_zero and os.name != 'nt':
        print(f"\n{ModelSummary(model, max_depth=1)}\n")
    
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
            
    if model_precision:
        cast_precision(model, model_precision)
        cast_precision(model.conditioner, model_precision)
        model.first_stage_model.to(torch.float32)

    fabric.barrier()
    torch.cuda.empty_cache()        
    
    trainer = Trainer(fabric, model, optimizer, scheduler, dataset, dataloader)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
