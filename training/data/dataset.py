"""Seeded, on-the-fly (mic, target) pairs and phase-2 loop components (DESIGN.md §1.3).

Every pair is a pure function of its integer seed, so batches at step s are fixed
by (run seed, s) and resuming is bit-exact regardless of worker scheduling.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve

from training.data import corpus, idcache, loop_sim, rir, synth

FS = 48000
PAIR_SEED_BASE = {"train": 10**12, "heldout": 9 * 10**12}
VAL_SEED_BASE = 99_000_000


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


class PairGenerator:
    """Builds training/validation material for one split ('train' | 'heldout')."""

    def __init__(self, cfg: dict, split: str):
        self.cfg, self.split = cfg, split
        self.sim = cfg["sim"]
        self.T = int(round(cfg["audio"]["seg_seconds"] * FS))
        self.source = cfg["data"]["source"]
        self.use_cache = split == "train" and bool(cfg["data"].get("id_cache", 0))
        self._ready = False
        self._cache = None

    def _lazy(self):                     # open files inside the worker process
        if self._ready:
            return
        if self.source == "real":
            self.speech_root, self.utts = corpus.speech_index(self.split)
            self.noise = corpus.noise_segments(self.split, self.cfg["data"]["noise_train_seconds"])
            self.bank = rir.Bank(self.split)
            self.n_scenes = min(self.bank.n, self.cfg["data"]["rir_train" if self.split == "train" else "rir_heldout"])
        else:
            self.n_scenes = self.cfg["data"]["rir_train" if self.split == "train" else "rir_heldout"]
            self._scene_cache = {}
        self._ready = True

    # ---------------------------------------------------------------- sources
    def scene(self, rng):
        i = int(rng.integers(self.n_scenes))
        if self.source == "real":
            return self.bank.scene(i)
        if i not in self._scene_cache:
            s = rir.make_scene(rir.SEEDS[self.split] + 500_000 + i, small=True)
            self._scene_cache[i] = dict(h_pa=s["h_pa"].astype(np.float32) * s["pa_gain"],
                                        h_talk=s["h_talk"].astype(np.float32), n_dir=s["n_dir"],
                                        msg_db=s["msg_db"], index=i)
        return self._scene_cache[i]

    def speech(self, rng, n_pre: int, T: int | None = None):
        """Speech of length n_pre + T (the pre-roll feeds the reverb tail into the crop)."""
        T = T or self.T
        n = n_pre + T
        if self.source != "real":
            return synth.speech(rng, n)
        for _ in range(20):
            u = self.utts[int(rng.integers(len(self.utts)))]
            start = int(rng.integers(0, max(1, u.frames - n + 1)))
            x, _ = sf.read(self.speech_root / u.path, start=start, frames=n, dtype="float32", always_2d=False)
            if len(x) < n:
                x = np.pad(x, (0, n - len(x)))
            if _rms(x[n_pre:]) > 1e-4:
                return x
        return x

    def speech_concat(self, rng, n: int):
        """n samples of one speaker's utterances back to back (eval scenes are 10 s long)."""
        if self.source != "real":
            return synth.speech(rng, n)
        spk = self.utts[int(rng.integers(len(self.utts)))].speaker
        pool = [u for u in self.utts if u.speaker == spk]
        parts, have = [], 0
        while have < n:
            u = pool[int(rng.integers(len(pool)))]
            x, _ = sf.read(self.speech_root / u.path, dtype="float32", always_2d=False)
            parts.append(x); have += len(x)
        return np.concatenate(parts)[:n]

    def noise_seg(self, rng, T: int | None = None):
        T = T or self.T
        if self.source != "real":
            return synth.noise(rng, T)
        path, a, b = self.noise[int(rng.integers(len(self.noise)))]
        start = int(rng.integers(a, b - T))
        x, _ = sf.read(path, start=start, frames=T, dtype="float32", always_2d=False)
        return x

    # ------------------------------------------------------------- one scene
    def components(self, seed: int, phase: int = 1, T: int | None = None, fixed: dict | None = None):
        """All loop ingredients for one example. `fixed` overrides sampled values (eval)."""
        self._lazy()
        fixed = fixed or {}
        T = T or self.T
        rng = np.random.default_rng(seed)
        sc = self.scene(rng) if "scene" not in fixed else fixed["scene"]
        h_talk, h_pa = sc["h_talk"], sc["h_pa"]
        n_pre = len(h_talk)
        # Long simulations (6 s identity pairs, phase-2 rollouts, 10 s eval) concatenate one speaker's
        # utterances: a single crop would often run past the utterance end into zero padding.
        concat = fixed.get("concat", T > int(2.5 * FS))
        x = self.speech_concat(rng, n_pre + T) if concat else self.speech(rng, n_pre, T)
        early = h_talk[: sc["n_dir"] + int(self.sim["early_ms"] * FS / 1000)].copy()
        f = min(240, len(early) // 4)
        early[len(early) - f:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, f)))
        s_mic = fftconvolve(x, h_talk)[n_pre:n_pre + T].astype(np.float32)
        s_tgt = fftconvolve(x, early)[n_pre:n_pre + T].astype(np.float32)
        level_db = fixed.get("level_db", rng.uniform(*self.sim["input_level_dbfs"]))
        k = 10 ** (level_db / 20) / _rms(s_mic)
        s_mic *= k; s_tgt *= k
        v = self.noise_seg(rng, T)
        snr = fixed.get("snr_db", rng.uniform(*self.sim["snr_db"]))
        v = (v * (_rms(s_mic) / (_rms(v) + 1e-12)) * 10 ** (-snr / 20)).astype(np.float32)
        hr = fixed.get("headroom_db", rng.uniform(*self.sim["headroom_db"]))
        c = _rms(s_mic) * 10 ** (hr / 20)
        if phase == 2:
            lo, hi = self.cfg["train"]["phase2"]["g_rel_db"]; p_above = self.cfg["train"]["phase2"]["p_above_msg"]
            mode = "model"
        else:
            mode = "id" if rng.random() < self.sim["p_id"] else "tf"
            mode = fixed.get("mode", mode)
            lo, hi = self.sim["id_g_rel_db"] if mode == "id" else self.sim["g_rel_db"]
            p_above = 1.0 if mode == "id" else self.sim["p_above_msg"]
        if mode == "id":
            g_rel = rng.uniform(lo, hi)
        else:
            g_rel = rng.uniform(0, hi) if rng.random() < p_above else rng.uniform(lo, 0)
        g_rel = fixed.get("g_rel_db", g_rel)
        dmin, dmax = (int(round(ms * FS / 1000)) for ms in self.sim["pa_delay_ms"])
        d = int(fixed.get("d", rng.integers(dmin, dmax + 1)))
        return dict(src=(s_mic + v).astype(np.float32), s_mic=s_mic, v=v, target=s_tgt, h_pa=h_pa.astype(np.float32),
                    G=float(10 ** (g_rel / 20)), g_rel=float(g_rel), d=d, c=float(c), mode=mode, scene=int(sc["index"]))

    def pair(self, seed: int, fixed: dict | None = None):
        """(mic, target, meta) for phase 1: teacher-forced or identity closed loop.

        Identity pairs are simulated for `sim.id_sim_seconds` (howl build-up in real rooms takes
        seconds) and the T-sample window ending at the guard trip is kept (the last T samples
        when the loop never trips). DESIGN §9 B1-1.
        """
        self._lazy()
        fixed = dict(fixed or {})
        rng = np.random.default_rng(seed + 7)
        mode = fixed.get("mode", "id" if rng.random() < self.sim["p_id"] else "tf")
        if mode == "id" and "mode" not in fixed and self.use_cache:
            if self._cache is None:
                self._cache = idcache.IdCache(self.cfg)
            mic, tgt, i = self._cache.draw(rng)
            return mic, tgt, dict(mode="id", cached=i, g_rel=float("nan"), tripped=False, trip_at=-1)
        fixed["mode"] = mode
        T = self.T
        T_sim = T if mode == "tf" else max(T, int(round(self.sim.get("id_sim_seconds", 0) * FS)) // 96 * 96)
        cp = self.components(seed, phase=1, T=T_sim, fixed=fixed)
        tripped, trip_at, w0 = False, -1, 0
        if mode == "tf":
            mic = loop_sim.open_loop_tf(cp["src"], cp["target"], cp["h_pa"], cp["G"], cp["d"], cp["c"])
        else:
            with torch.no_grad():
                r = loop_sim.run_loop(cp["src"][None], cp["h_pa"][None], [cp["G"]], [cp["d"]], [cp["c"]],
                                      loop_sim.Identity(), self.sim["guard_clipped"], self.sim["guard_window"])
            mic = r.mic[0].numpy()
            tripped, trip_at = bool(r.tripped[0]), int(r.trip_at[0])
            end = trip_at + FS // 10 if tripped else T_sim          # keep 100 ms of saturation onset
            w0 = int(min(max(end, T), T_sim) - T)
        meta = dict(mode=mode, g_rel=cp["g_rel"], d=cp["d"], scene=cp["scene"], tripped=tripped,
                    trip_at=trip_at - w0 if tripped else -1)
        return mic[w0:w0 + T].astype(np.float32), cp["target"][w0:w0 + T].copy(), meta


class PairDataset(torch.utils.data.Dataset):
    """Map-style view: item i is the pair with seed base + i (phase 1) or loop components (phase 2)."""

    def __init__(self, cfg: dict, split: str, base_seed: int, length: int, phase: int = 1):
        self.gen = PairGenerator(cfg, split)
        self.base, self.length, self.phase = base_seed, length, phase

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        seed = self.base + int(i)
        if self.phase == 1:
            mic, tgt, meta = self.gen.pair(seed)
            return torch.from_numpy(mic), torch.from_numpy(tgt), torch.tensor(float(meta["mode"] == "id"))
        T2 = int(round(self.gen.cfg["train"]["phase2"].get("rollout_seconds", self.gen.cfg["audio"]["seg_seconds"]) * FS)) // 96 * 96
        cp = self.gen.components(seed, phase=2, T=T2)
        return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v))
                for k, v in cp.items() if k not in ("mode",)}


class ValDataset(torch.utils.data.Dataset):
    """The fixed held-out validation pairs (module level so spawned DataLoader workers can pickle it)."""

    def __init__(self, cfg: dict, n: int):
        self.gen, self.n = PairGenerator(cfg, "heldout"), n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        mic, tgt, meta = self.gen.pair(VAL_SEED_BASE + int(i))
        return torch.from_numpy(mic), torch.from_numpy(tgt), torch.tensor(float(meta["mode"] == "id"))


def collate_phase2(items):
    """Pad RIRs to a common length; stack everything else."""
    L = max(len(it["h_pa"]) for it in items)
    out = {k: torch.stack([it[k] for it in items]) for k in items[0] if k != "h_pa"}
    out["h_pa"] = torch.stack([torch.nn.functional.pad(it["h_pa"], (0, L - len(it["h_pa"]))) for it in items])
    return out


def worker_init(_):
    torch.set_num_threads(1)


def train_seed_base(cfg: dict, phase: int) -> int:
    return PAIR_SEED_BASE["train"] + cfg["seed"] * 10**9 + (phase - 1) * 5 * 10**8


def bench(cfg: dict, workers: int = 3, n_batches: int = 12, batch: int = 16) -> float:
    """Pairs per second from a DataLoader with `workers` processes (B1 throughput check)."""
    import time
    ds = PairDataset(cfg, "train", train_seed_base(cfg, 1), 10**9)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, worker_init_fn=worker_init,
                                     persistent_workers=False, prefetch_factor=4 if workers else None)
    it = iter(dl)
    next(it)                                   # warm-up (worker start, file opening)
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    return n_batches * batch / (time.time() - t0)


if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/s.yaml")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.bench:
        torch.set_num_threads(1)
        print(f"throughput: {bench(yaml.safe_load(open(a.config)), a.workers):.1f} pairs/s with {a.workers} workers")
