"""Eval metrics (DESIGN.md §6): SI-SDR, STOI (16 kHz, pystoi), howl-energy ratio, level offset."""
from __future__ import annotations

import numpy as np
import torch
from pystoi import stoi as _stoi
from scipy.signal import resample_poly

from training.model.loss import howl_ratio

FS = 48000


def sisdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - est.mean(); ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    t = a * ref
    return float(10 * np.log10(np.dot(t, t) / max(np.dot(est - t, est - t), eps) + eps))


def stoi(est: np.ndarray, ref: np.ndarray) -> float:
    return float(_stoi(resample_poly(ref, 1, 3), resample_poly(est, 1, 3), 16000, extended=False))


def howl_energy(est: np.ndarray, ref: np.ndarray) -> float:
    with torch.no_grad():
        return float(howl_ratio(torch.tensor(est[None], dtype=torch.float32), torch.tensor(ref[None], dtype=torch.float32))[0])


def level_offset_db(est: np.ndarray, ref: np.ndarray) -> float:
    return float(10 * np.log10((np.mean(est.astype(np.float64) ** 2) + 1e-20) / (np.mean(ref.astype(np.float64) ** 2) + 1e-20)))
