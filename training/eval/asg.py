"""Added stable gain via the Python loop simulator (DESIGN.md §6, CLOUD-TRAINING.md §7 B4).

MSG(proc) per held-out scene = largest g_rel (dB re the identity MSG estimate) at
which a 10 s closed-loop run is stable: the saturation guard never trips and the
feedback-to-source ratio over the last 2 s is <= fsr_max_db. Binary search with
all scenes batched. ASG = MSG(model) - MSG(identity).
"""
from __future__ import annotations

import numpy as np
import torch

from training.data import loop_sim
from training.data.dataset import PairGenerator

FS = 48000


class Scenes:
    """Fixed held-out eval scenes: bank scene i, concatenated test-clean speech, fixed level/SNR/headroom."""

    def __init__(self, cfg: dict, n: int | None = None, seconds: float | None = None, indices=None):
        ev = cfg["eval"]
        indices = list(range(n or ev["scenes"])) if indices is None else list(indices)
        self.indices, self.n = indices, len(indices)
        self.T = int(round((seconds or ev["seconds"]) * FS)) // 96 * 96
        gen = PairGenerator(cfg, "heldout")
        gen._lazy()
        self.items = []
        for i in indices:
            scene = gen.bank.scene(i) if gen.source == "real" else gen.scene(np.random.default_rng(i))
            fixed = {"scene": scene, "level_db": ev["input_level_dbfs"], "snr_db": ev["snr_db"],
                     "headroom_db": ev["headroom_db"], "g_rel_db": 0.0, "concat": True}
            self.items.append(gen.components(2_000_000 + i, phase=2, T=self.T, fixed=fixed))
        L = max(len(c["h_pa"]) for c in self.items)
        self.src = torch.tensor(np.stack([c["src"] for c in self.items]))
        self.target = np.stack([c["target"] for c in self.items])
        self.h_pa = torch.tensor(np.stack([np.pad(c["h_pa"], (0, L - len(c["h_pa"]))) for c in self.items]))
        self.d = [c["d"] for c in self.items]
        self.c = [c["c"] for c in self.items]
        self.fsr_max = ev["fsr_max_db"]
        self.guard = (cfg["sim"]["guard_clipped"], cfg["sim"]["guard_window"])

    def run(self, proc, g_rel_db) -> loop_sim.LoopResult:
        G = 10 ** (np.asarray(g_rel_db, dtype=np.float64) / 20)
        with torch.no_grad():
            return loop_sim.run_loop(self.src, self.h_pa, G, self.d, self.c, proc, *self.guard)

    def stable(self, proc, g_rel_db) -> np.ndarray:
        r = self.run(proc, g_rel_db)
        q = 2 * FS
        fb = (r.mic[:, -q:] - self.src[:, -q:]).double()
        fsr = 10 * torch.log10((fb ** 2).mean(1) / ((self.src[:, -q:].double() ** 2).mean(1) + 1e-20) + 1e-20)
        return (~r.tripped).numpy() & (fsr.numpy() <= self.fsr_max)


def msg_search(scenes: Scenes, proc, lo: float = -12.0, hi: float = 24.0, iters: int = 9):
    """Per-scene binary search; returns (msg_db array, flags) where flags mark clamped results."""
    n = scenes.n
    lo_a, hi_a = np.full(n, lo), np.full(n, hi)
    ok_lo = scenes.stable(proc, lo_a)
    ok_hi = scenes.stable(proc, hi_a)
    for _ in range(iters):
        mid = (lo_a + hi_a) / 2
        s = scenes.stable(proc, mid)
        lo_a = np.where(s, mid, lo_a); hi_a = np.where(s, hi_a, mid)
    msg = np.where(ok_hi, hi, np.where(ok_lo, lo_a, lo))
    flags = np.where(ok_hi, "above-range", np.where(ok_lo, "", "below-range"))
    return msg, flags


def asg(scenes: Scenes, proc_factory, ident_factory=loop_sim.Identity, lo=-12.0, hi=24.0, iters=9) -> dict:
    m_id, f_id = msg_search(scenes, ident_factory(), lo, hi, iters)
    m_pr, f_pr = msg_search(scenes, proc_factory(), lo, hi, iters)
    a = m_pr - m_id
    return {"msg_identity": m_id, "msg_proc": m_pr, "asg": a, "flags_identity": f_id, "flags_proc": f_pr,
            "asg_median": float(np.median(a)), "asg_p10": float(np.percentile(a, 10))}
