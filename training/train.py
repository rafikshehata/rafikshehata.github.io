"""Chunked, resumable training loop (DESIGN.md §5; CLOUD-TRAINING.md §7 B3, §8).

    uv run python train.py --config configs/s.yaml --run s-v1 --resume --max-minutes 25
    uv run python train.py --config configs/s.yaml --run probe --probe [--probe-phase 2]
    uv run python train.py --config configs/s.yaml --run s-v1 --resume --max-minutes 0   # unlimited (local)

Step-based: the data of step s is a pure function of (seed, s), so resuming from
latest.pt is bit-exact. Checkpoints (atomic) every `ckpt_minutes` and at exit:
latest.pt, prev.pt (the previous latest), best.pt (best val SI-SDR) + state.json.
Exits 0 when the minute budget is reached, on SIGTERM, or when training is done.

`train.procs` > 1 runs that many single-threaded data-parallel ranks (gloo
all-reduce) splitting the global batch; on this 4-vCPU VM one thread is nearly
as fast as four for these small GRUs, so processes use the cores far better
(DESIGN.md §9 B3-2). Rank 0 owns logging and checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

from training.data import dataset as D
from training.data import loop_sim
from training.model import loss as L
from training.model.net import RinglessNet
from training.model.streaming import Streaming

ROOT = pathlib.Path(__file__).resolve().parent
_STOP = {"flag": False}


def _on_term(signum, frame):
    _STOP["flag"] = True


def lr_at(cfg: dict, step: int) -> float:
    t = cfg["train"]; p1 = t["phase1_steps"]
    if step < p1:
        if step < t["warmup"]:
            return t["lr"] * (step + 1) / t["warmup"]
        prog = min(1.0, (step - t["warmup"]) / max(1, p1 - t["warmup"]))
        return t["lr_min"] + 0.5 * (t["lr"] - t["lr_min"]) * (1 + math.cos(math.pi * prog))
    p2 = t["phase2"]
    prog = min(1.0, (step - p1) / max(1, t["phase2_steps"]))
    return p2["lr_min"] + 0.5 * (p2["lr"] - p2["lr_min"]) * (1 + math.cos(math.pi * prog))


def make_opt(net: RinglessNet, cfg: dict, lr: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = name.endswith(("weight_ih_l0", "weight_hh_l0")) or name == "dense.weight"
        (decay if is_matrix else no_decay).append(p)        # filterbank A/S: no decay (DESIGN §9 B3-1)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def save_atomic(obj, path: pathlib.Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json_atomic(obj, path: pathlib.Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


class RankSampler(torch.utils.data.Sampler):
    """Dataset indices for one rank: global step s owns items [s·B, (s+1)·B); rank r takes its slice."""

    def __init__(self, first_step_item: int, global_batch: int, rank: int, world: int):
        self.first, self.B, self.rank, self.world = first_step_item, global_batch, rank, world
        self.b = global_batch // world

    def __iter__(self):
        s = 0
        while True:
            base = self.first + s * self.B + self.rank * self.b
            yield from range(base, base + self.b)
            s += 1

    def __len__(self):
        return 10**12


class Trainer:
    def __init__(self, args, rank: int = 0, world: int = 1):
        self.args, self.rank, self.world = args, rank, world
        self.is0 = rank == 0
        self.cfg = cfg = yaml.safe_load(open(args.config))
        self.cfg_path = str(args.config)
        self.dir = ROOT / "runs" / args.run
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1 if world > 1 else cfg["train"]["threads"])
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        self.t = cfg["train"]
        for B in (self.t["batch"], self.t["phase2"]["batch"]):
            assert B % world == 0, f"batch {B} not divisible by {world} ranks"
        self.p1, self.p2 = self.t["phase1_steps"], self.t["phase2_steps"]
        self.net = RinglessNet.from_config(cfg)
        self.model = self.net
        if world > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.net)
        self.opt = make_opt(self.net, cfg, lr_at(cfg, 0))
        self.step, self.p2_started = 0, False
        self.best, self.best_step = -float("inf"), -1
        self.chunks, self.train_seconds, self.nan_steps = 0, 0.0, 0
        self.last_val_step, self.last_val = -1, None
        if args.resume and (self.dir / "latest.pt").exists():
            self.load(self.dir / "latest.pt")
        self.metrics = open(self.dir / "metrics.jsonl", "a") if self.is0 else None
        self._val = None

    def say(self, msg: str) -> None:
        if self.is0:
            print(msg, flush=True)

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {"version": 1, "model": self.net.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                "phase": self.phase, "p2_started": self.p2_started, "best_val_sisdr": self.best,
                "best_step": self.best_step, "chunks": self.chunks, "train_seconds": self.train_seconds,
                "nan_steps": self.nan_steps, "last_val_step": self.last_val_step, "last_val": self.last_val,
                "config": self.cfg, "config_path": self.cfg_path, "world": self.world,
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}

    def load(self, path: pathlib.Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        for k in ("step", "p2_started", "best_step", "chunks", "train_seconds", "nan_steps", "last_val_step", "last_val"):
            setattr(self, k, ck[k])
        self.best = ck["best_val_sisdr"]
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"])
        self.chunks += 1
        adam = [int(s["step"]) for s in self.opt.state_dict()["state"].values()][:1]
        if ck.get("world", 1) != self.world:
            self.say(f"note: checkpoint written with {ck.get('world', 1)} ranks, resuming with {self.world}")
        self.say(f"resumed at step {self.step} (phase {self.phase}); optimizer state restored "
                 f"(adam_step={adam[0] if adam else 0})")

    @property
    def phase(self) -> int:
        return 1 if self.step < self.p1 else 2

    def checkpoint(self, reason: str) -> None:
        if not self.is0:
            return
        latest, prev = self.dir / "latest.pt", self.dir / "prev.pt"
        if latest.exists():
            os.replace(latest, prev)
        save_atomic(self.state_dict(), latest)
        self.write_state(reason)

    def write_state(self, reason: str = "") -> None:
        if not self.is0:
            return
        write_json_atomic({"run": self.args.run, "config": self.cfg_path, "step": self.step, "phase": self.phase,
                           "phase1_steps": self.p1, "phase2_steps": self.p2,
                           "done": self.step >= self.p1 + self.p2, "best_val_sisdr": self.best,
                           "best_step": self.best_step, "last_val": self.last_val, "chunks": self.chunks,
                           "train_seconds": round(self.train_seconds, 1), "nan_steps": self.nan_steps,
                           "world": self.world, "reason": reason,
                           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                          self.dir / "state.json")

    def log(self, rec: dict) -> None:
        if self.is0:
            self.metrics.write(json.dumps(rec) + "\n"); self.metrics.flush()

    def allreduce(self, values: list[float], op=None) -> list[float]:
        if self.world == 1:
            return values
        t = torch.tensor(values, dtype=torch.float64)
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
        return t.tolist()

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    # ------------------------------------------------------------------ data
    def loader(self, phase: int, start_step: int):
        B = self.t["batch"] if phase == 1 else self.t["phase2"]["batch"]
        first = (start_step - (0 if phase == 1 else self.p1)) * B
        ds = D.PairDataset(self.cfg, "train", D.train_seed_base(self.cfg, phase), 10**12, phase=phase)
        w = self.cfg["data"]["workers"]
        dl = torch.utils.data.DataLoader(
            ds, batch_size=B // self.world, sampler=RankSampler(first, B, self.rank, self.world), num_workers=w,
            worker_init_fn=D.worker_init, collate_fn=D.collate_phase2 if phase == 2 else None,
            prefetch_factor=4 if w else None, persistent_workers=False)
        return iter(dl)

    def val_set(self):
        if self._val is not None:
            return self._val
        n, T = self.t["val_pairs"], int(self.cfg["audio"]["seg_seconds"] * 48000)
        cache = ROOT / "data_cache" / f"val_{self.cfg['name']}_{self.cfg['data']['source']}_{n}_{T}.pt"
        if self.is0 and not cache.exists():
            w = self.cfg["data"]["workers"]
            parts = list(torch.utils.data.DataLoader(D.ValDataset(self.cfg, n), batch_size=10, num_workers=w,
                                                     worker_init_fn=D.worker_init))
            data = tuple(torch.cat([p[k] for p in parts]) for k in range(3))
            cache.parent.mkdir(exist_ok=True)
            save_atomic(data, cache)
        self.barrier()
        self._val = torch.load(cache, weights_only=False)
        return self._val

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def validate(self) -> dict:
        t0 = time.time()
        mic, tgt, is_id = self.val_set()
        idx = torch.arange(self.rank, len(mic), self.world)
        sums = np.zeros(8)                  # loss·n, n, si, si_in, si_tf, n_tf, si_id, n_id
        for i in range(0, len(idx), 25):
            j = idx[i:i + 25]
            y = self.net(mic[j])
            loss, _ = L.total_loss(y, tgt[j], self.cfg["loss"])
            si, si_in = L.sisdr(y, tgt[j]), L.sisdr(mic[j], tgt[j])
            si = torch.nan_to_num(si, nan=-100.0, posinf=-100.0, neginf=-100.0)
            m_id = is_id[j] == 1
            sums += [float(loss) * len(j), len(j), float(si.sum()), float(si_in.sum()), float(si[~m_id].sum()),
                     float((~m_id).sum()), float(si[m_id].sum()), float(m_id.sum())]
        sums = np.array(self.allreduce(sums.tolist()))
        rec = {"step": self.step, "phase": self.phase, "val_loss": sums[0] / sums[1], "val_sisdr": sums[2] / sums[1],
               "val_sisdr_in": sums[3] / sums[1], "val_sisdr_tf": sums[4] / max(sums[5], 1),
               "val_sisdr_id": sums[6] / max(sums[7], 1), "val_seconds": round(time.time() - t0, 1)}
        if not math.isfinite(rec["val_sisdr"]):
            rec["val_sisdr"] = -1e9
        improved = rec["val_sisdr"] > self.best
        if improved:
            self.best, self.best_step = rec["val_sisdr"], self.step
        rec.update(best_val_sisdr=self.best, best_step=self.best_step, elapsed=round(self.train_seconds, 1))
        self.last_val_step, self.last_val = self.step, {k: rec[k] for k in ("val_loss", "val_sisdr", "val_sisdr_in")}
        self.log(rec)
        if improved and self.is0:
            save_atomic(self.state_dict(), self.dir / "best.pt")
        self.say(f"val step {self.step}: sisdr {rec['val_sisdr']:.2f} dB (input {rec['val_sisdr_in']:.2f}) "
                 f"loss {rec['val_loss']:.4f} best {self.best:.2f}@{self.best_step}")
        self.barrier()
        return rec

    # -------------------------------------------------------------- training
    def start_phase2(self) -> None:
        if (self.dir / "best.pt").exists():
            ck = torch.load(self.dir / "best.pt", map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"])
            self.say(f"phase 2: initialised from best phase-1 checkpoint (step {ck['step']}, "
                     f"val SI-SDR {ck['best_val_sisdr']:.2f} dB)")
        self.opt = make_opt(self.net, self.cfg, lr_at(self.cfg, self.step))
        self.p2_started = True

    def train_step(self, batch) -> dict:
        info = {}
        if self.phase == 1:
            mic, tgt, is_id = batch
            info["id_frac"] = float(is_id.mean())
        else:
            s = self.cfg["sim"]
            with torch.no_grad():
                r = loop_sim.run_loop(batch["src"], batch["h_pa"], batch["G"], batch["d"], batch["c"],
                                      Streaming(self.net), s["guard_clipped"], s["guard_window"])
            mic, tgt = r.mic, batch["target"]
            info["trip_rate"] = float(r.tripped.float().mean())
        y = self.model(mic, tbptt=self.t["phase2"]["tbptt_frames"] if self.phase == 2 else None)
        loss, parts = L.total_loss(y, tgt, self.cfg["loss"])
        info.update(parts)
        self.opt.zero_grad(set_to_none=True)
        finite = self.allreduce([float(torch.isfinite(loss.detach()))], dist.ReduceOp.MIN if self.world > 1 else None)[0]
        if not finite:                                          # every rank skips together
            self.nan_steps += 1
            info.update(loss=float(loss.detach()), grad_norm=float("nan"), skipped=1.0)
            return info
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.t["grad_clip"])
        self.opt.step()
        info.update(loss=float(loss.detach()), grad_norm=float(gn))
        return info

    def run(self) -> int:
        a = self.args
        t_start = time.time()
        budget = a.max_minutes * 60 if a.max_minutes > 0 else float("inf")
        total = self.p1 + self.p2
        if a.max_steps:
            total = min(total, a.max_steps)
        if self.step >= total:
            self.say(f"training already done ({self.step}/{total} steps)")
            self.write_state("done"); return 0
        self.write_state("start")
        reserve = 30.0                                     # final validation + checkpoint
        last_ckpt = time.time()
        it, it_phase = None, None
        window, w_t0, w_steps = [], time.time(), 0
        while True:
            if self.phase == 2 and not self.p2_started:
                if self.last_val_step != self.step:
                    self.validate()
                self.start_phase2()
            if it_phase != self.phase:
                it, it_phase = self.loader(self.phase, self.step), self.phase
            batch = next(it)
            lr = lr_at(self.cfg, self.step)
            for g in self.opt.param_groups:
                g["lr"] = lr
            t0 = time.time()
            info = self.train_step(batch)
            self.step += 1
            self.train_seconds += time.time() - t0
            window.append(info); w_steps += 1
            if self.step % self.t["log_every"] == 0 or self.step == total:
                dt = time.time() - w_t0
                keys = [k for k in window[0] if all(isinstance(w.get(k), (int, float)) for w in window)]
                means = self.allreduce([float(np.mean([w[k] for w in window])) for k in keys])
                rec = {"step": self.step, "phase": self.phase, "lr": lr, "steps_per_s": round(w_steps / dt, 3),
                       "elapsed": round(self.train_seconds, 1)}
                rec.update({k: v / self.world for k, v in zip(keys, means)})
                self.log(rec)
                self.say(f"step {self.step} ph{self.phase} loss {rec.get('loss', float('nan')):.4f} "
                         f"sisdr {rec.get('sisdr', float('nan')):.2f} lr {lr:.2e} {rec['steps_per_s']:.3f} it/s")
                window, w_t0, w_steps = [], time.time(), 0
            val_every = self.t["val_every"] if self.phase == 1 else self.t.get("val_every_phase2", self.t["val_every"])
            if self.step % val_every == 0:
                v0 = time.time(); self.validate(); reserve = max(reserve, 1.5 * (time.time() - v0) + 10)
            if a.stop_at_step and self.step >= a.stop_at_step:
                _STOP["flag"] = True
            want_ckpt = time.time() - last_ckpt >= self.t["ckpt_minutes"] * 60
            want_stop = _STOP["flag"] or self.step >= total or time.time() - t_start >= budget - reserve
            flags = self.allreduce([float(want_ckpt), float(want_stop), float(_STOP["flag"])],
                                   dist.ReduceOp.MAX if self.world > 1 else None)
            if flags[2]:
                _STOP["flag"] = True
            if flags[1]:
                break
            if flags[0]:
                self.checkpoint("periodic"); last_ckpt = time.time()
        if self.last_val_step != self.step:
            self.validate()
        reason = "sigterm/stop" if _STOP["flag"] else ("done" if self.step >= total else "budget")
        self.checkpoint(reason)
        self.say(f"exit: {reason} at step {self.step} after {time.time() - t_start:.0f} s")
        return 0

    def probe(self, phase: int, seconds: float) -> int:
        """Measure steps/s for `seconds` (no checkpoints). Phase 2 uses the current weights in the loop."""
        if phase == 2:
            self.step = self.p1; self.p2_started = True
        it = self.loader(phase, self.step)
        for _ in range(3):                                   # warm-up: workers, caches
            self.train_step(next(it))
        self.barrier()
        n, t0 = 0, time.time()
        while True:
            self.train_step(next(it)); n += 1
            if self.allreduce([float(time.time() - t0 >= seconds)], dist.ReduceOp.MAX if self.world > 1 else None)[0]:
                break
        sps = n / (time.time() - t0)
        rec = {"probe_phase": phase, "steps": n, "steps_per_s": round(sps, 4),
               "steps_per_25min_chunk": int(sps * (25 * 60 - 90)), "procs": self.world,
               "threads": torch.get_num_threads(), "workers_per_proc": self.cfg["data"]["workers"],
               "batch": self.t["batch"] if phase == 1 else self.t["phase2"]["batch"]}
        self.say("PROBE " + json.dumps(rec))
        return 0


def _rank_main(rank: int, world: int, args, init: str) -> None:
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        tr = Trainer(args, rank, world)
        code = tr.probe(args.probe_phase, args.probe_seconds) if args.probe else tr.run()
    finally:
        dist.destroy_process_group()
    if code:
        sys.exit(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=25.0, help="0 = unlimited (local mode)")
    ap.add_argument("--probe", action="store_true", help="run ~2 minutes and print steps/s")
    ap.add_argument("--probe-phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--probe-seconds", type=float, default=120.0)
    ap.add_argument("--procs", type=int, default=None, help="override train.procs")
    ap.add_argument("--stop-at-step", type=int, default=0, help="testing: checkpoint and exit after this step")
    ap.add_argument("--max-steps", type=int, default=0, help="testing: cap the total number of steps")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    world = a.procs or cfg["train"].get("procs", 1)
    if world <= 1:
        signal.signal(signal.SIGTERM, _on_term)
        tr = Trainer(a)
        return tr.probe(a.probe_phase, a.probe_seconds) if a.probe else tr.run()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    ctx = torch.multiprocessing.start_processes(_rank_main, args=(world, a, f"tcp://127.0.0.1:{port}"),
                                                nprocs=world, join=False, start_method="spawn")
    def forward(signum, frame):                               # pass SIGTERM on to every rank
        for p in ctx.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        while not ctx.join():
            pass
    except Exception as e:                                    # a rank failed
        print(f"rank failure: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
