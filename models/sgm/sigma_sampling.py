import torch

from common.utils import get_class
from .encoder_util import default


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization, num_idx=1000, do_append_zero=False, flip=True, **kwargs):
        self.num_idx = num_idx
        self.discretization = get_class(discretization)
        self.sigmas = self.discretization()(num_idx, do_append_zero=do_append_zero, flip=flip)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            torch.randint(0, self.num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
