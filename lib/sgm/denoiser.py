import torch
import torch.nn as nn
from .encoder_util import append_dims

from abc import abstractmethod
from functools import partial

import numpy as np
import torch

from .model_util import make_beta_schedule
from .encoder_util import append_zero, default

def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2.0


class EpsScaling:
    def __call__(self, sigma):
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            "linear", num_timesteps, linear_start=linear_start, linear_end=linear_end
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def get_sigmas(self, n, device="cpu"):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))
    

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.weighting = EpsWeighting()
        self.scaling = EpsScaling()

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network, input, sigma, cond):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        num_idx=1000,
        do_append_zero=False,
        quantize_c_noise=True,
        flip=True,
    ):
        super().__init__()
        sigmas = LegacyDDPMDiscretization()(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma):
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma):
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise):
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise


class DiscreteSampling:
    def __init__(self, num_idx=1000, do_append_zero=False, flip=True):
        self.num_idx = num_idx
        self.sigmas = LegacyDDPMDiscretization()(num_idx, do_append_zero=do_append_zero, flip=flip)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)