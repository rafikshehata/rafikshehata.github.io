"""Closed-loop PA feedback simulator (DESIGN.md §1.2).

    mic[n] = src[n] + (h_pa * spk)[n],   spk[n] = clip_c(G * p[n - d]),   p = proc(mic)

`src` is everything at the mic except feedback (talker + noise). The loop runs
block-wise (partition B = 96 samples) with uniformly partitioned overlap-save
FFT convolution; d >= B makes every block computable from the past. Batched in
torch so the same code serves data generation (identity processor), phase-2
training and eval (streaming model processor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import fftconvolve

BLOCK = 96
STREAM_DELAY = 48          # model output stream delay (W - H), DESIGN §2


def msg_db(h: np.ndarray) -> float:
    n = 1 << int(math.ceil(math.log2(max(4 * len(h), 2))))
    return float(-20 * np.log10(np.abs(np.fft.rfft(h, n)).max()))


class PartitionedConv:
    """Batched uniformly-partitioned overlap-save convolution of a block stream with fixed FIRs.

    h: (B, L) float tensor. step(x) takes (B, block) and returns (h * x) for that block.
    """

    def __init__(self, h: torch.Tensor, block: int = BLOCK):
        Bt, L = h.shape
        self.block = block
        self.P = max(1, math.ceil(L / block))
        hp = torch.nn.functional.pad(h, (0, self.P * block - L)).reshape(Bt, self.P, block)
        H = torch.fft.rfft(torch.cat([hp, torch.zeros_like(hp)], -1), dim=-1)      # (B, P, K)
        # Reversed partitions, laid out (B, K, P) so each block is one contiguous dot product.
        self.Hrev = H.flip(1).transpose(1, 2).contiguous()
        K = block + 1
        self.fdl = torch.zeros(Bt, K, 2 * self.P, dtype=H.dtype)                # doubled ring
        self.prev = torch.zeros(Bt, block, dtype=h.dtype)
        self.pos = 0

    def step(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(torch.cat([self.prev, x], -1), dim=-1)                 # (B, K)
        self.prev = x
        self.fdl[:, :, self.pos] = X
        self.fdl[:, :, self.pos + self.P] = X
        s = (self.pos + 1) % self.P
        win = self.fdl[:, :, s:s + self.P]                                        # oldest .. newest
        Y = (win.unsqueeze(-2) @ self.Hrev.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        self.pos = (self.pos + 1) % self.P
        return torch.fft.irfft(Y, n=2 * self.block, dim=-1)[:, self.block:]


class Identity:
    def reset(self, batch: int):
        pass

    def step(self, x):
        return x


class Gain:
    """Fixed linear gain processor (tests: a -6 dB 'model')."""

    def __init__(self, gain: float):
        self.gain = gain

    def reset(self, batch: int):
        pass

    def step(self, x):
        return self.gain * x


@dataclass
class LoopResult:
    mic: torch.Tensor          # (B, T)
    p: torch.Tensor            # (B, T) processor output stream
    spk: torch.Tensor          # (B, T) loudspeaker signal (after clip / mute)
    tripped: torch.Tensor      # (B,) bool
    trip_at: torch.Tensor      # (B,) long, -1 if never


def run_loop(src, h_pa, G, d, c, proc, guard_clipped: int = 100, guard_window: int = 480,
             mute_on_trip: bool = True, block: int = BLOCK) -> LoopResult:
    """Run the closed loop for T = src.shape[1] samples (T multiple of `block`).

    src (B,T), h_pa (B,L) normalised RIR, G (B,) linear gain, d (B,) delay in samples (>= block),
    c (B,) clip level. Guard: >= guard_clipped clipped samples within guard_window samples trips;
    after a trip the PA is muted when mute_on_trip (protection mute, DESIGN §1.2 / §9 D1).
    """
    src = torch.as_tensor(src, dtype=torch.float32)
    Bt, T = src.shape
    assert T % block == 0, "T must be a multiple of the block size"
    h_pa = torch.as_tensor(h_pa, dtype=torch.float32)
    G = torch.as_tensor(G, dtype=torch.float32).reshape(Bt, 1)
    c = torch.as_tensor(c, dtype=torch.float32).reshape(Bt, 1)
    d = torch.as_tensor(d, dtype=torch.long).reshape(Bt)
    assert int(d.min()) >= block, "PA delay must be at least one partition"
    dmax = int(d.max())
    conv = PartitionedConv(h_pa, block)
    proc.reset(Bt)
    pfull = torch.zeros(Bt, dmax + T)
    mic = torch.zeros(Bt, T); spk_all = torch.zeros(Bt, T)
    clip_hist = torch.zeros(Bt, guard_window)
    tripped = torch.zeros(Bt, dtype=torch.bool)
    trip_at = torch.full((Bt,), -1, dtype=torch.long)
    ar = torch.arange(block)
    for j in range(T // block):
        n0 = j * block
        idx = (dmax + n0 + ar).unsqueeze(0) - d.unsqueeze(1)                       # p[n - d]
        g = G * torch.gather(pfull, 1, idx)
        clipped = (g.abs() >= c).float()
        spk = torch.minimum(torch.maximum(g, -c), c)
        # guard: clipped samples in the trailing window, evaluated per sample of this block
        hist = torch.cat([clip_hist, clipped], 1)
        cs = torch.cumsum(hist, 1)
        cnt = cs[:, guard_window:] - cs[:, :block]                                 # window ending at each sample
        hit = cnt >= guard_clipped
        new = hit.any(1) & ~tripped
        if new.any():
            first = hit.float().argmax(1)
            trip_at = torch.where(new, n0 + first, trip_at)
            tripped = tripped | new
        clip_hist = hist[:, block:]
        if mute_on_trip and tripped.any():
            live = (n0 + ar).unsqueeze(0) < torch.where(tripped, trip_at, torch.full_like(trip_at, 1 << 62)).unsqueeze(1)
            spk = spk * live
        m = src[:, n0:n0 + block] + conv.step(spk)
        p = proc.step(m)
        pfull[:, dmax + n0:dmax + n0 + block] = p
        mic[:, n0:n0 + block] = m
        spk_all[:, n0:n0 + block] = spk
    return LoopResult(mic, pfull[:, dmax:], spk_all, tripped, trip_at)


def open_loop_tf(src: np.ndarray, target: np.ndarray, h_pa: np.ndarray, G: float, d: int, c: float) -> np.ndarray:
    """Teacher-forced mic: the PA plays the clipped, delayed target (p[n] = s_tgt[n - 48])."""
    T = len(src)
    p = np.zeros(T, np.float32)
    lag = d + STREAM_DELAY
    if lag < T:
        p[lag:] = target[:T - lag]
    spk = np.clip(G * p, -c, c).astype(np.float32)
    fb = fftconvolve(spk, h_pa)[:T].astype(np.float32)
    return (src + fb).astype(np.float32)


def energy_db(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    return 10 * np.log10(np.mean(x ** 2, axis=axis) + 1e-20)
