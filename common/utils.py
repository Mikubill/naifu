import torch
import os
import safetensors.torch
import torch.utils._device
from lightning.pytorch.utilities import rank_zero_only
from common.logging import logger


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    logger.info(*args, **kwargs)


@rank_zero_only
def rank_zero_warn(*args, **kwargs):
    logger.warning(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


class EmptyInitWrapper(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None):
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        return func(*args, **kwargs)


def load_torch_file(ckpt, safe_load=False, device=None):
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device)
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:
                print(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
                )
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            rank_zero_print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def setup_smddp(config):
    from lightning.pytorch.plugins.environments import LightningEnvironment

    # from lightning.fabric.strategies import DDPStrategy
    from common.fairscale import DDPShardedStrategy

    env = LightningEnvironment()
    env.world_size = lambda: int(os.environ["WORLD_SIZE"])
    env.global_rank = lambda: int(os.environ["RANK"])
    strategy = DDPShardedStrategy(
        cluster_environment=env,
        accelerator="gpu",
        static_graph=True,
    )

    world_size = int(os.environ["WORLD_SIZE"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_nodes = int(world_size / num_gpus)
    init_params = {
        "devices": num_gpus,
        "num_nodes": num_nodes,
    }

    config.lightning.update(init_params)
    del config.lightning.accelerator

    config.trainer.checkpoint_dir = os.path.join(
        "/opt/ml/checkpoints", config.trainer.checkpoint_dir
    )
    config.sampling.save_dir = os.path.join(
        os.environ.get("SM_OUTPUT_DIR"), config.sampling.save_dir
    )
    return strategy, config


def get_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    items = sorted(os.listdir(checkpoint_dir))
    if not items:
        return None
    return os.path.join(checkpoint_dir, items[-1])
