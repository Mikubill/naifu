
import math
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from data.buckets import AspectRatioSampler
from data.store import AspectRatioDataset, ImageStore
from pytorch_lightning.utilities import rank_zero_only
from torch_ema import ExponentialMovingAverage
from lib.utils import get_local_rank, get_world_size
from lib.utils import min_snr_weighted_loss, load_torch_file
from omegaconf import OmegaConf 
from pathlib import Path
from diffusers import DDIMScheduler
from lib.sgm import GeneralConditioner
from lib.wrappers import AutoencoderKLWrapper, UnetWrapper
        
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# define the LightningModule
class StableDiffusionModel(pl.LightningModule):
    def __init__(self, model_path, config, batch_size):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.batch_size = batch_size 
        self.lr = self.config.optimizer.params.get("lr", 1e-4)
        self.save_hyperparameters(config)
        self.init_model()
        self.automatic_optimization = False
        
    def init_model(self):
        config = self.config
        
        # assume path is file, we get the config from the file
        yaml_file = Path(self.model_path).with_suffix(".yaml")
        if not yaml_file.is_file():
            # assume it's sdxl
            yaml_file = Path("lib/model_configs/sd_xl_base.yaml")
        self.model_config = OmegaConf.load(yaml_file)
        model_params = self.model_config.model.params
        
        for item in model_params.conditioner_config.params.emb_models:
            item["target"] = item["target"].replace("modules.", "")
            item["target"] = "lib." + item["target"]   
        
        if self.config.trainer.use_xformers:
            model_params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
            model_params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers"
        
        encoder = AutoencoderKLWrapper(**model_params.first_stage_config.params).eval()
        encoder.train = disabled_train
        for param in encoder.parameters():
            param.requires_grad = False
        self.first_stage_model = encoder
        self.scale_factor = model_params.scale_factor

        self.model = UnetWrapper(model_params.network_config.params)
        self.conditioner = GeneralConditioner(**model_params.conditioner_config.params)
        self.noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        
        print(f"Loading model from {self.model_path}")
        sd = load_torch_file(self.model_path)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
            
        self.cast_dtype = torch.float32
        self.conditioner.to(torch.float16)    
        if config.trainer.use_ema: 
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
            
        # self.use_latent_cache = self.config.dataset.get("cache_latents")
            
    def setup(self, stage):
        local_rank = get_local_rank()
        world_size = get_world_size()
        dataset_cls = AspectRatioDataset if self.config.arb.enabled else ImageStore
        
        # init Dataset
        self.dataset = dataset_cls(
            size=self.config.trainer.resolution,
            seed=self.config.trainer.seed,
            rank=local_rank,
            init=not self.config.arb.enabled,
            **self.config.dataset,
            **self.config.cache
        )
        
        # init sampler
        self.data_sampler = None
        if self.config.arb.enabled:
            self.data_sampler = AspectRatioSampler(
                bsz=self.batch_size,
                config=self.config, 
                rank=local_rank, 
                dataset=self.dataset, 
                world_size=world_size,
            ) 
        self.disable_amp()
            
    def disable_amp(self):
        if self.cast_dtype == torch.float32:
            return
        from pytorch_lightning.plugins import PrecisionPlugin
        precision_plugin = PrecisionPlugin()
        precision_plugin.precision = self.config.lightning.precision
        self.trainer.strategy.precision_plugin = precision_plugin

    def train_dataloader(self):
        if self.data_sampler:
            self.data_sampler.update_bsz(self.batch_size)
            
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collate_fn,
            sampler=self.data_sampler,
            num_workers=self.config.dataset.num_workers,
            batch_size=1 if self.data_sampler else self.batch_size,
            persistent_workers=True,
        )
        return dataloader
    
    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=False):
            out = self.first_stage_model._decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=False):
            z = self.first_stage_model._encode(x)
        z = self.scale_factor * z
        return z
    
    def setup_cache(self):
        if self.config.get("cache") is None:
            return
        
        if self.config.cache.get("enabled") == True:
            self.dataset.setup_cache(self.encode_first_stage, self.conditioner, self.data_sampler.buckets)
            self.conditioner.to("cpu")
            self.first_stage_model.to("cpu")
            torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):  
        if "latents" not in batch.keys():
            latents = self.encode_first_stage(batch["images"])
            if torch.any(torch.isnan(latents)):
                print("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
                
            del batch["images"]
            cond = self.conditioner(batch)
        else:
            latents = batch["latents"]
            cond = batch["conds"]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.config.trainer.get("offset_noise"):
            noise = torch.randn_like(latents) + float(self.config.trainer.get("offset_noise_val")) \
                * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
        
        # https://arxiv.org/abs/2301.11706
        if self.config.trainer.get("input_perturbation"):
            noise = noise + float(self.config.trainer.get("input_perturbation_val")) * torch.randn_like(noise)

        bsz = latents.shape[0]
            
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
        # Predict the noise residual
        noise_pred = self.model(noisy_latents, timesteps, cond)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if not self.config.trainer.get("min_snr"):
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")  
        else:
            gamma = self.config.trainer.get("min_snr_val")
            loss = min_snr_weighted_loss(noise_pred.float(), target.float(), timesteps, self.noise_scheduler, gamma=gamma)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")
            
        optimizers = self.optimizers()
        optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        current_step = self.trainer.global_step
        if (current_step + 1) % accumulate_grad_batches == 0:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)
        if (current_step + 1) % accumulate_grad_batches == 0:
            for opt in optimizers:
                opt.step()
                
        # Logging to TensorBoard by default
        major, minor, _ = pl.__version__.split('.')
        if int(major) >= 2:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("train_loss", loss)
    
    def get_scaled_lr(self, base):
        # Scale LR OPs
        f = self.trainer.accumulate_grad_batches * self.config.trainer.batch_size * self.trainer.num_nodes * self.trainer.num_devices
        if self.config.trainer.lr_scale == "linear":
            return base * f, True
        elif self.config.trainer.lr_scale == "sqrt":
            return base * math.sqrt(f), True
        elif self.config.trainer.lr_scale == "none":
            return base, False
        else:
            raise ValueError(self.config.lr_scale)
    
    def configure_optimizers(self):
        # align LR with batch size in case we're using lrfinder
        if self.lr != self.config.optimizer.params.get("lr", self.lr):
            self.config.optimizer.params.lr = self.lr
            
        new_lr, scaled = self.get_scaled_lr(self.lr)
        if scaled and self.config.optimizer.params.get("lr") != None:
            self.config.optimizer.params.lr = new_lr
            rank_zero_only(print(f"Using scaled LR: {self.config.optimizer.params.lr}"))
            
        params_to_optim = [{'params': self.model.diffusion_model.parameters()}]
        if self.config.trainer.get("train_text_encoder") == True:
            text_encoder_group = {'params': self.text_encoder.parameters()}
            if self.config.trainer.get("text_encoder_lr"):
                text_encoder_group['lr'], te_scaled = self.get_scaled_lr(self.config.trainer.get("text_encoder_lr"))
            params_to_optim.append(text_encoder_group)

        optimizer = get_class(self.config.optimizer.name)(
            params_to_optim, **self.config.optimizer.params
        )
        scheduler = get_class(self.config.lr_scheduler.name)(
            optimizer=optimizer,
            **self.config.lr_scheduler.params
        )
        
        warmup_config = self.config.lr_scheduler.warmup
        if warmup_config.enabled and self.trainer.global_step < warmup_config.num_warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = min(pg["lr"], warmup_config.init_lr)
            
        return [[optimizer], [scheduler]]
    
    def lr_scheduler_step(self, *args):
        warmup_config = self.config.lr_scheduler.warmup
        if not warmup_config.enabled or self.trainer.global_step > warmup_config.num_warmup:
            super().lr_scheduler_step(*args)
                
    def optimizer_step(self, epoch, batch_idx, optimizer, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, *args, **kwargs)
        
        warmup_config = self.config.lr_scheduler.warmup
        if warmup_config.enabled and self.trainer.global_step < warmup_config.num_warmup:
            f = min(1.0, float(self.trainer.global_step + 1) / float(warmup_config.num_warmup))
            if warmup_config.strategy == "cos":
                f = (math.cos(math.pi*(1+f))+1)/2.
            delta = self.config.optimizer.params.lr-warmup_config.init_lr
            for pg in optimizer.param_groups:
                if pg["lr"] >= warmup_config.init_lr:
                    pg["lr"] = warmup_config.init_lr+f*delta
    
    def on_train_start(self):
        if self.config.trainer.use_ema: 
            self.ema.to(self.device, dtype=self.unet.dtype)
            
        self.setup_cache()
            
    # def on_train_epoch_start(self) -> None:
    #     if self.use_latent_cache:
    #         self.vae.to("cpu")
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.config.trainer.use_ema:
            self.ema.update()
            
    def on_save_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            checkpoint["model_ema"] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint):
        if self.config.trainer.use_ema:
            self.ema.load_state_dict(checkpoint["model_ema"])



def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)
