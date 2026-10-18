"""Write WAV pairs for listening: input (mic, ringing) vs target (clean).

    uv run python -m training.data.preview --n 5 [--config configs/s.yaml] [--out runs/preview]

Pairs use the identity closed loop with g_rel in [+0.5, +3] dB above the MSG
estimate, so the input rings or howls; files are float WAV at 48 kHz.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import soundfile as sf
import yaml

from training.data.dataset import PairGenerator
from training.data.loop_sim import energy_db

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--config", default=str(ROOT / "configs" / "s.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "preview"))
    ap.add_argument("--source", choices=["real", "synthetic"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.source:
        cfg["data"]["source"] = a.source
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gen = PairGenerator(cfg, "heldout")
    rng = np.random.default_rng(7)
    for i in range(a.n):
        g = float(rng.uniform(0.5, 3.0))
        mic, tgt, meta = gen.pair(77_000 + i, fixed={"mode": "id", "g_rel_db": g})
        sf.write(out / f"pair{i}_input.wav", mic, 48000, subtype="FLOAT")
        sf.write(out / f"pair{i}_target.wav", tgt, 48000, subtype="FLOAT")
        q = len(mic) // 4
        print(f"pair{i}: g_rel {g:+.1f} dB  tripped {meta['tripped']} at {meta['trip_at']}  "
              f"input-target {energy_db(mic) - energy_db(tgt):+.1f} dB  "
              f"input max-quarter vs first-quarter {max(energy_db(mic[k*q:(k+1)*q]) for k in range(4)) - energy_db(mic[:q]):+.1f} dB")
    print(f"wrote {a.n} pairs to {out}")


if __name__ == "__main__":
    main()
