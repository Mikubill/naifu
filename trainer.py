# python trainer.py --model_path=/tmp/model --config config/test.yaml
import os

# Hide welcome message from bitsandbytes
os.environ.update({
    "BITSANDBYTES_NOWELCOME": "1",
    "DIFFUSERS_VERBOSITY": "error"
})

import torch
import lightning.pytorch as pl

from lib.args import parse_args
from lib.callbacks import HuggingFaceHubCallback, SampleCallback
from lib.model import StableDiffusionModel, get_pipeline
from lib.compat import pl_compat_fix
from lib.precision import HalfPrecisionPlugin

from omegaconf import OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch import seed_everything
from lib.utils import rank_zero_print

def main(args):
    config = OmegaConf.load(args.config)
    seed_everything(config.trainer.seed)

    if args.model_path == None:
        args.model_path = config.trainer.model_path

    strategy = None
    if config.lightning.accelerator in ["gpu", "cpu"]:
        strategy = "ddp"

    if config.trainer.use_hivemind:
        from lib.hivemind import init_hivemind
        strategy = init_hivemind(config)

    rank_zero_print(f"Loading model from {args.model_path}")
    pipeline = get_pipeline(args.model_path)
    if config.get("lora"):
        if config.lora.get("use_locon"):
            from experiment.locon import LoConDiffusionModel
            model = LoConDiffusionModel(pipeline, config)
        else:
            from experiment.lora import LoRADiffusionModel
            model = LoRADiffusionModel(pipeline, config)
        strategy = config.lightning.strategy = "auto"
    else:
        model = StableDiffusionModel(pipeline, config)

    major, minor = torch.__version__.split('.')[:2]
    if (int(major) > 1 or (int(major) == 1 and int(minor) >= 12)) and torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{device.major}.{device.minor}")
        precision = 'high' if config.lightning.precision == 32 else 'medium'
        if compute_capability >= 8.0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

    callbacks = []
    if config.monitor.huggingface_repo != "":
        hf_logger = HuggingFaceHubCallback(
            repo_name=config.monitor.huggingface_repo,
            use_auth_token=config.monitor.hf_auth_token,
            **config.monitor
        )
        callbacks.append(hf_logger)

    logger = None
    if config.monitor.wandb_id != "":
        logger = WandbLogger(project=config.monitor.wandb_id)
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    if config.get("custom_embeddings") != None and config.custom_embeddings.enabled:
        from experiment.textual_inversion import CustomEmbeddingsCallback
        callbacks.append(CustomEmbeddingsCallback(config.custom_embeddings))

    if config.get("sampling") != None and config.sampling.enabled:
        callbacks.append(SampleCallback(config.sampling, logger))

    if torch.cuda.device_count() == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")

    if config.lightning.get("strategy") is not None:
        strategy = config.lightning.strategy
        del config.lightning["strategy"]

    if not config.get("custom_embeddings") or not config.custom_embeddings.freeze_unet:
        checkpoint_config = {
            k: v
            for k, v in config.checkpoint.items() if k != "extended"
        }
        callbacks.append(ModelCheckpoint(**checkpoint_config))
        enable_checkpointing = True
    else:
        enable_checkpointing = False

    if config.lightning.get("enable_checkpointing") == None:
        config.lightning.enable_checkpointing = enable_checkpointing
        
    plugins = None
    target_precision = config.lightning.precision
    if target_precision in ["16-true", "bf16-true"]:
        plugins = HalfPrecisionPlugin(target_precision)
        model.to(torch.float16 if target_precision == "16-true" else torch.bfloat16)
        del config.lightning.precision

    # config.lightning.replace_sampler_ddp = False
    config, callbacks = pl_compat_fix(config, callbacks)
    trainer = pl.Trainer(
        logger=logger, 
        callbacks=callbacks, 
        strategy=strategy, 
        plugins=plugins, 
        **config.lightning
    )
    trainer.fit(model=model, ckpt_path=args.resume if args.resume else None)

if __name__ == "__main__":
    args = parse_args()
    main(args)

