"""Room impulse responses: pyroomacoustics shoebox ISM + cached bank (DESIGN.md §1.1).

    uv run python -m training.data.rir --build      # train + held-out banks (idempotent)

Each scene has a talker->mic RIR and a PA->mic RIR, both peak-normalised and
stored as float16 in flat .npy arrays (memory-mapped by the data workers).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, sosfilt

from training.data.fetch import data_dir

FS = 48000
SEEDS = {"train": 0, "heldout": 1_000_000}


def _fade_truncate(h: np.ndarray, max_len: int) -> np.ndarray:
    e = np.cumsum(h[::-1] ** 2)[::-1]
    edc = 10 * np.log10(e / (e[0] + 1e-30) + 1e-30)
    below = np.nonzero(edc < -60)[0]
    n = int(min(below[0] if below.size else len(h), max_len, len(h)))
    h = h[:n].copy()
    f = min(240, n // 4)                                          # 5 ms half-Hann fade-out
    h[n - f:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, f)))
    return h


def msg_db(h: np.ndarray) -> float:
    """Identity-processor MSG estimate: -20 log10 max_f |H(f)| (DESIGN §1.2)."""
    n = 1 << int(math.ceil(math.log2(max(4 * len(h), 2))))
    return float(-20 * np.log10(np.abs(np.fft.rfft(h, n)).max()))


def _place(rng, dims, z_range, margin):
    lo = np.array([margin, margin, z_range[0]])
    hi = np.array([dims[0] - margin, dims[1] - margin, z_range[1]])
    return lo + rng.random(3) * (hi - lo)


def make_scene(seed: int, small: bool = False) -> dict:
    """One room with PA and talker RIRs. `small=True` gives fast rooms for the synthetic source."""
    rng = np.random.default_rng(seed)
    if small:
        V, rt60 = float(np.exp(rng.uniform(np.log(40), np.log(150)))), float(rng.uniform(0.2, 0.4))
    else:
        V, rt60 = float(np.exp(rng.uniform(np.log(40), np.log(2000)))), float(rng.uniform(0.2, 1.2))
    Hr = float(rng.uniform(2.7, np.clip(2.7 + V / 400, 3.2, 8.0)))
    A = V / Hr
    rho = rng.uniform(1.0, 2.0)
    L = float(np.sqrt(A * rho)); Wd = A / L
    dims = [L, Wd, Hr]
    rt_used = rt60
    for _ in range(40):
        try:
            e_abs, max_order = pra.inverse_sabine(rt_used, dims)
            break
        except ValueError:
            rt_used *= 1.1
    e_abs = min(e_abs, 0.99)
    max_order = int(min(max_order, 12 if small else 100))

    mic = _place(rng, dims, (1.2, min(2.0, Hr - 0.3)), 0.5)
    diag = float(np.linalg.norm(dims))
    r_target = rng.uniform(1.0, max(1.0, min(12.0, 0.8 * diag)))
    best, best_err = None, 1e9
    for _ in range(300):
        pa = _place(rng, dims, (1.8, min(3.0, Hr - 0.3)), 0.5)
        err = abs(np.linalg.norm(pa - mic) - r_target)
        if err < best_err:
            best, best_err = pa, err
        if err < 0.1:
            break
    pa = best
    r_talk = rng.uniform(0.05, 0.6)
    for _ in range(100):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        talker = mic + r_talk * v
        if np.all(talker > 0.3) and np.all(talker < np.array(dims) - 0.3):
            break
    else:
        talker = np.clip(talker, 0.3, np.array(dims) - 0.3)

    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=max_order, air_absorption=False)
    room.add_source(pa.tolist()); room.add_source(talker.tolist()); room.add_microphone(mic.tolist())
    room.compute_rir()
    h_pa = np.asarray(room.rir[0][0], dtype=np.float64)
    h_talk = np.asarray(room.rir[0][1], dtype=np.float64)

    hp = rng.uniform(80, 150); lp = rng.uniform(12000, 18000)
    h_pa = sosfilt(butter(2, hp, "highpass", fs=FS, output="sos"), h_pa)
    h_pa = sosfilt(butter(4, lp, "lowpass", fs=FS, output="sos"), h_pa)
    h_pa = _fade_truncate(h_pa, FS)                    # <= 1.0 s
    h_talk = _fade_truncate(h_talk, FS // 2)           # <= 0.5 s
    raw_msg = msg_db(h_pa)
    h_pa = (h_pa / np.abs(h_pa).max()).astype(np.float16)
    h_talk = (h_talk / np.abs(h_talk).max()).astype(np.float16)
    pa_gain = float(10 ** (msg_db(h_pa.astype(np.float32)) / 20))    # makes max|H_pa| = 1
    return dict(h_pa=h_pa, h_talk=h_talk, pa_gain=pa_gain, msg_db=raw_msg,
                n_dir=int(np.argmax(np.abs(h_talk))),
                meta=dict(seed=seed, V=V, rt60=rt60, rt60_used=rt_used, dims=dims, max_order=max_order,
                          d_pa=float(np.linalg.norm(pa - mic)), d_talk=float(r_talk), hp=hp, lp=lp))


def bank_dir():
    d = data_dir() / "rir"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_bank(split: str, n: int, workers: int | None = None) -> None:
    d = bank_dir()
    if (d / f"{split}.json").exists() and json.loads((d / f"{split}.json").read_text())["n"] == n:
        print(f"rir: {split} bank already built ({n} scenes)")
        return
    t0 = time.time()
    seeds = [SEEDS[split] + i for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:
        scenes = list(pool.map(make_scene, seeds, chunksize=8))
    for key in ("h_pa", "h_talk"):
        lens = np.array([len(s[key]) for s in scenes])
        off = np.concatenate([[0], np.cumsum(lens)])
        np.save(d / f"{split}_{key}.npy", np.concatenate([s[key] for s in scenes]))
        np.save(d / f"{split}_{key}_off.npy", off)
    np.save(d / f"{split}_pa_gain.npy", np.array([s["pa_gain"] for s in scenes], np.float32))
    np.save(d / f"{split}_n_dir.npy", np.array([s["n_dir"] for s in scenes], np.int64))
    np.save(d / f"{split}_msg_db.npy", np.array([s["msg_db"] for s in scenes], np.float32))
    info = dict(n=n, seeds=[seeds[0], seeds[-1]], seconds=round(time.time() - t0, 1),
                meta=[s["meta"] for s in scenes])
    (d / f"{split}.json").write_text(json.dumps(info))
    size = sum(p.stat().st_size for p in d.glob(f"{split}_*.npy"))
    print(f"rir: built {split} bank: {n} scenes in {info['seconds']} s, {size / 1e6:.0f} MB")


class Bank:
    """Memory-mapped access to a built RIR bank."""

    def __init__(self, split: str):
        d = bank_dir()
        self.n = json.loads((d / f"{split}.json").read_text())["n"]
        self._a = {k: np.load(d / f"{split}_{k}.npy", mmap_mode="r") for k in ("h_pa", "h_talk")}
        self._o = {k: np.load(d / f"{split}_{k}_off.npy") for k in ("h_pa", "h_talk")}
        self.pa_gain = np.load(d / f"{split}_pa_gain.npy")
        self.n_dir = np.load(d / f"{split}_n_dir.npy")
        self.msg_db = np.load(d / f"{split}_msg_db.npy")

    def scene(self, i: int) -> dict:
        get = lambda k: np.asarray(self._a[k][self._o[k][i]:self._o[k][i + 1]], dtype=np.float32)
        return dict(h_pa=get("h_pa") * self.pa_gain[i], h_talk=get("h_talk"),
                    n_dir=int(self.n_dir[i]), msg_db=float(self.msg_db[i]), index=i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-heldout", type=int, default=200)
    a = ap.parse_args()
    if a.build:
        build_bank("train", a.n_train)
        build_bank("heldout", a.n_heldout)


if __name__ == "__main__":
    main()
