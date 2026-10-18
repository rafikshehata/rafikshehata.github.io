"""Learnable real filterbank front-end (DESIGN.md §2): framing, analysis, bin pairing, synthesis, OLA."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

W, H = 96, 48
N_BINS = W // 2 + 1                                   # 49


def sqrt_hann(n: int = W) -> np.ndarray:
    return np.sqrt(0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n)))


def dft_rows(n: int = W) -> np.ndarray:
    """Real DFT basis: rows 0..n/2 = cos(2πkn/N); rows n/2+1..n-1 = -sin(2πkn/N), k = 1..n/2-1."""
    t = np.arange(n)
    cos = np.cos(2 * np.pi * np.outer(np.arange(n // 2 + 1), t) / n)
    sin = -np.sin(2 * np.pi * np.outer(np.arange(1, n // 2), t) / n)
    return np.concatenate([cos, sin], 0)


def analysis_init(n: int = W) -> np.ndarray:
    return dft_rows(n) * sqrt_hann(n)[None, :]


def synthesis_init(n: int = W) -> np.ndarray:
    """S0 = diag(sqrt-Hann) · inv(DFT rows): S0·A0·x = Hann ⊙ x, which overlap-adds to x at hop n/2."""
    return sqrt_hann(n)[:, None] * np.linalg.inv(dft_rows(n))


def bin_rows(n: int = W) -> list[list[int]]:
    """Rows of the coefficient vector that belong to each of the n/2+1 bins."""
    half = n // 2
    return [[0]] + [[k, half + k] for k in range(1, half)] + [[half]]


def pair_power(c: torch.Tensor) -> torch.Tensor:
    """(..., 96) coefficients -> (..., 49) per-bin power."""
    half = c.shape[-1] // 2
    p = c[..., : half + 1] ** 2
    return torch.cat([p[..., :1], p[..., 1:half] + c[..., half + 1:] ** 2, p[..., half:]], -1)


def expand_mask(m: torch.Tensor) -> torch.Tensor:
    """(..., 49) bin mask -> (..., 96) per-row mask."""
    return torch.cat([m, m[..., 1:-1]], -1)


def frames(x: torch.Tensor, w: int = W, h: int = H) -> torch.Tensor:
    """(B, T) -> (B, T/h + 1, w). Frame t covers x[t·h - (w-h) : t·h + h]; one extra frame closes the OLA."""
    assert x.shape[-1] % h == 0
    return F.pad(x, (w - h, h)).unfold(-1, w, h)


def overlap_add(z: torch.Tensor, T: int, w: int = W, h: int = H) -> torch.Tensor:
    """(B, Tf, w) frames -> (B, T) signal aligned with the input of `frames` (inverse framing)."""
    Bt, Tf, _ = z.shape
    out = F.fold(z.transpose(1, 2), output_size=(1, (Tf - 1) * h + w), kernel_size=(1, w), stride=(1, h))
    return out.reshape(Bt, -1)[:, w - h: w - h + T]
