import torch
import math
from packaging import version
from lib.utils import get_free_memory
from lib.sgm import UNetModel, AutoencoderKL

# code from https://github.com/comfyanonymous/ComfyUI/blob/a9a4ba7574c06ce9364afaffb09d0435b5ff19c9/comfy/model_management.py#L418
class AutoencoderKLWrapper(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UnetWrapper(torch.nn.Module):
    def __init__(self, config, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        diffusion_model = UNetModel(**config)
        self.diffusion_model = compile(diffusion_model)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs
        )
        