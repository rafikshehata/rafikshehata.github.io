"""Synthetic sources so tests and the tiny smoke run need no downloaded data (DESIGN.md §1.3)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

FS = 48000
_LP8K = butter(8, 7800, "lowpass", fs=FS, output="sos")


def _resonator(f, bw):
    r = np.exp(-np.pi * bw / FS)
    return [1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r]


def speech(rng: np.random.Generator, n: int) -> np.ndarray:
    """Speech-like signal: pitched syllables with moving formants, fricatives and pauses, <= 8 kHz."""
    out = np.zeros(n)
    f0_base = rng.uniform(90, 220)
    t = int(rng.integers(0, FS // 10))
    while t < n:
        if rng.random() < 0.25:                                    # pause
            t += int(rng.uniform(0.05, 0.3) * FS)
            continue
        L = int(rng.uniform(0.1, 0.3) * FS)
        L = min(L, n - t)
        if L <= 16:
            break
        if rng.random() < 0.2:                                     # fricative
            seg = sosfilt(butter(2, rng.uniform(2500, 5000), "highpass", fs=FS, output="sos"),
                          rng.normal(size=L)) * 0.3
        else:                                                      # voiced syllable
            f0 = f0_base * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(2, 5) * np.arange(L) / FS + rng.uniform(0, 6)))
            phase = np.cumsum(f0 / FS)
            exc = np.diff(np.floor(phase), prepend=0.0) - np.mean(np.diff(np.floor(phase), prepend=0.0))
            seg = exc
            for fc, bw in ((rng.uniform(300, 800), 80), (rng.uniform(900, 2200), 100), (rng.uniform(2200, 3200), 150)):
                b, a = _resonator(fc, bw)
                seg = lfilter(b, a, seg)
        env = np.sin(np.pi * np.arange(L) / L) ** 2
        out[t:t + L] += seg * env * rng.uniform(0.5, 1.0)
        t += L
    out = sosfilt(_LP8K, out)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Pink-ish noise with slow level modulation (venue-like hum + hiss)."""
    w = rng.normal(size=n)
    pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
    mod = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 1.0) * np.arange(n) / FS)
    x = pink * mod
    return (x / (np.std(x) + 1e-9)).astype(np.float32)
