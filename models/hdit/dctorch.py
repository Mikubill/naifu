import torch
import scipy.fft
import numpy as np
from functools import lru_cache

# dctorch/funcitonal.py


@lru_cache()
def compute_dct_mat(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = scipy.fft.dct(np.eye(n), norm="ortho")
    return torch.tensor(m, device=device, dtype=dtype)


@lru_cache()
def compute_idct_mat(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    m = scipy.fft.idct(np.eye(n), norm="ortho")
    return torch.tensor(m, device=device, dtype=dtype)


def dct(t: torch.Tensor) -> torch.Tensor:
    m = compute_dct_mat(t.shape[-2], device=t.device, dtype=t.dtype)
    return torch.einsum("...id,ij->...jd", t, m)


def idct(t: torch.Tensor) -> torch.Tensor:
    m = compute_idct_mat(t.shape[-2], device=t.device, dtype=t.dtype)
    return torch.einsum("...id,ij->...jd", t, m)


def dct2(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    mh = compute_dct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_dct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...hw,hi,wj->...ij", t, mh, mw)


def idct2(t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]
    mh = compute_idct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_idct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...hw,hi,wj->...ij", t, mh, mw)


def dct3(t: torch.Tensor) -> torch.Tensor:
    l, h, w = t.shape[-3:]
    ml = compute_dct_mat(l, device=t.device, dtype=t.dtype)
    mh = compute_dct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_dct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...lhw,li,hj,wk->...ijk", t, ml, mh, mw)


def idct3(t: torch.Tensor) -> torch.Tensor:
    l, h, w = t.shape[-3:]
    ml = compute_idct_mat(l, device=t.device, dtype=t.dtype)
    mh = compute_idct_mat(h, device=t.device, dtype=t.dtype)
    mw = compute_idct_mat(w, device=t.device, dtype=t.dtype)
    return torch.einsum("...lhw,li,hj,wk->...ijk", t, ml, mh, mw)
