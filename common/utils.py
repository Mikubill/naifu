from typing import Callable, Optional
import torch
import os
import safetensors.torch
import torch.utils._device
import argparse
import sys

from common.logging import logger
from tqdm import tqdm

from transformers.utils import logging as tsl
from diffusers.utils import logging as dsl

tsl.disable_default_handler()
tsl.get_logger().addHandler(logger.handlers[0])


dsl.disable_default_handler()
dsl.get_logger().addHandler(logger.handlers[0])


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def create_scaled_precision_plugin():
    from lightning.fabric.plugins.precision.amp import MixedPrecision
    scaler = torch.cuda.amp.GradScaler()
    org_unscale_grads = scaler._unscale_grads_
    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)
    scaler._unscale_grads_ = _unscale_grads_replacer
    return MixedPrecision("16-mixed", "cuda", scaler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    first_args = sys.argv[1]
    
    if first_args.startswith("--"):
        args = parser.parse_args()
    else:
        args = parser.parse_args(sys.argv[2:])
        args.config = first_args
    return args

class ProgressBar:
    def __init__(self, total: int, disable=False):
        if disable:
            self.progress = None
            return

        default_desc = "Epoch 0"
        self.is_rich_progress = False
        self.progress = tqdm(total=total, desc=default_desc)

    def update(self, desc: str, step: int, status: str = ""):
        if not self.progress:
            return

        if step == 0:
            self.progress.reset()

        self.progress.n = step
        self.progress.set_description_str(desc)
        self.progress.set_postfix_str(status)


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


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(
        self, lr: float = 1e-3, optimizer_dict: Optional[dict] = None, *args, **kwargs
    ) -> None:
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        pass


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


def load_torch_file(ckpt, safe_load=False, device=None, extract=True):
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

        sd = pl_sd
        if extract:
            if "global_step" in pl_sd:
                logger.info(f"Global Step: {pl_sd['global_step']}")
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]

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


def get_latest_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    items = sorted(os.listdir(checkpoint_dir))
    # remove all _optimizer.pt
    items = [x for x in items if "_optimizer" not in x]
    if not items:
        return None
    return os.path.join(checkpoint_dir, items[-1])
