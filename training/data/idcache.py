"""Precomputed identity-loop pairs (CLOUD-TRAINING.md §7 B1: "else add a precomputed cache").

Identity pairs need a 6 s closed-loop simulation each (~0.7 CPU-s), which halves
data throughput. They are generated once into $RL_DATA_DIR/idcache/ (float16,
per-pair peak scaling) and drawn by seed during phase 1 with a random ±6 dB gain
and polarity. Teacher-forced pairs stay fully on the fly.

    uv run python -m training.data.idcache --build --config configs/s.yaml [--n 4000]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import yaml

from training.data.fetch import data_dir

SEED_BASE = 700_000_000_000
_GEN = {}


def _one(args):
    cfg_text, seed = args
    if "gen" not in _GEN:
        from training.data.dataset import PairGenerator
        torch.set_num_threads(1)
        _GEN["gen"] = PairGenerator(yaml.safe_load(cfg_text), "train")
    mic, tgt, meta = _GEN["gen"].pair(seed, fixed={"mode": "id"})
    scale = float(max(np.abs(mic).max(), np.abs(tgt).max(), 1e-12))
    return (mic / scale).astype(np.float16), (tgt / scale).astype(np.float16), scale, meta


def cache_dir(cfg: dict):
    T = int(cfg["audio"]["seg_seconds"] * 48000)
    return data_dir() / "idcache" / f"{cfg['data']['source']}_{T}_{cfg['sim']['id_sim_seconds']}"


def build(cfg: dict, n: int, workers: int | None = None) -> None:
    d = cache_dir(cfg)
    if (d / "info.json").exists() and json.loads((d / "info.json").read_text())["n"] >= n:
        print(f"idcache: already built ({d})")
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T = int(cfg["audio"]["seg_seconds"] * 48000)
    mic = np.lib.format.open_memmap(d / "mic.npy.tmp", mode="w+", dtype=np.float16, shape=(n, T))
    tgt = np.lib.format.open_memmap(d / "tgt.npy.tmp", mode="w+", dtype=np.float16, shape=(n, T))
    scale, metas = np.zeros(n, np.float32), []
    text = yaml.safe_dump(cfg)
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:
        for i, (m, t, s, meta) in enumerate(pool.map(_one, [(text, SEED_BASE + i) for i in range(n)], chunksize=16)):
            mic[i], tgt[i], scale[i] = m, t, s
            metas.append(meta)
            if (i + 1) % 500 == 0:
                print(f"idcache: {i + 1}/{n} in {time.time() - t0:.0f} s", flush=True)
    mic.flush(); tgt.flush(); del mic, tgt
    os.replace(d / "mic.npy.tmp", d / "mic.npy"); os.replace(d / "tgt.npy.tmp", d / "tgt.npy")
    np.save(d / "scale.npy", scale)
    trips = float(np.mean([m["tripped"] for m in metas]))
    (d / "info.json").write_text(json.dumps({"n": n, "seed_base": SEED_BASE, "trip_rate": trips,
                                             "seconds": round(time.time() - t0, 1)}))
    print(f"idcache: built {n} pairs in {time.time() - t0:.0f} s (trip rate {trips:.2f}) at {d}")


class IdCache:
    def __init__(self, cfg: dict):
        d = cache_dir(cfg)
        self.n = json.loads((d / "info.json").read_text())["n"]
        self.mic = np.load(d / "mic.npy", mmap_mode="r")
        self.tgt = np.load(d / "tgt.npy", mmap_mode="r")
        self.scale = np.load(d / "scale.npy")

    def draw(self, rng: np.random.Generator):
        i = int(rng.integers(self.n))
        g = self.scale[i] * 10 ** (rng.uniform(-6, 6) / 20) * (1 if rng.random() < 0.5 else -1)
        return (self.mic[i].astype(np.float32) * g), (self.tgt[i].astype(np.float32) * g), i


def available(cfg: dict) -> bool:
    return (cache_dir(cfg) / "info.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--config", default="configs/s.yaml")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.build:
        build(cfg, a.n or cfg["data"].get("id_cache", 4000))


if __name__ == "__main__":
    main()
