import torch
import math
from packaging import version
from lib.utils import get_free_memory
from lib.sgm import UNetModel, AutoencoderKL

# code from https://github.com/comfyanonymous/ComfyUI/blob/a9a4ba7574c06ce9364afaffb09d0435b5ff19c9/comfy/model_management.py#L418
class AutoencoderKLWrapper(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_tiling = False
    
    @staticmethod
    def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
        return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

    @staticmethod
    @torch.inference_mode()
    def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, pbar = None):
        output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device="cpu")
        for b in range(samples.shape[0]):
            s = samples[b:b+1]
            out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
            out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
            for y in range(0, s.shape[2], tile_y - overlap):
                for x in range(0, s.shape[3], tile_x - overlap):
                    s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                    ps = function(s_in).cpu()
                    mask = torch.ones_like(ps)
                    feather = round(overlap * upscale_amount)
                    for t in range(feather):
                            mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                    out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                    out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                    if pbar is not None:
                        pbar.update(1)

            output[b:b+1] = out/out_div
        return output
    
    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap = 16):
        steps = samples.shape[0] * self.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * self.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * self.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        decode_fn = lambda a: (self.decode(a.to(self.dtype).to(self.device)) + 1.0).float()
        output = torch.clamp((
            (self.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = 8) +
            self.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = 8) +
             self.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount = 8))
            / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap = 64):
        steps = pixel_samples.shape[0] * self.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * self.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * self.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        encode_fn = lambda a: self.encode(2. * a.to(self.dtype).to(self.device) - 1.).sample().float()
        samples = self.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount = (1/8), out_channels=4)
        samples += self.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount = (1/8), out_channels=4)
        samples += self.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount = (1/8), out_channels=4)
        samples /= 3.0
        return samples

    def _decode(self, samples_in):
        free_memory = get_free_memory(self.device)
        batch_number = int((free_memory * 0.7) / (2562 * samples_in.shape[2] * samples_in.shape[3] * 64))
        batch_number = max(1, batch_number)
        pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * 8), round(samples_in.shape[3] * 8)), device="cpu")
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x+batch_number].to(self.dtype).to(self.device)
            pixel_samples[x:x+batch_number] = torch.clamp((self.decode(samples) + 1.0) / 2.0, min=0.0, max=1.0)
        return pixel_samples

    def _encode(self, pixel_samples):
        free_memory = get_free_memory(self.device)
        batch_number = int((free_memory * 0.7) / (2078 * pixel_samples.shape[2] * pixel_samples.shape[3])) #NOTE: this constant along with the one in the decode above are estimated from the mem usage for the VAE and could change.
        batch_number = max(1, batch_number)
        samples = torch.empty((pixel_samples.shape[0], 4, round(pixel_samples.shape[2] // 8), round(pixel_samples.shape[3] // 8)), device="cpu")
        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = (2. * pixel_samples[x:x+batch_number] - 1.).to(self.dtype).to(self.device)
            samples[x:x+batch_number] = self.encode(pixels_in).sample()
        return samples.to(self.device)


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
        