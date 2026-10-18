"""Evaluate a checkpoint against gate G2c and write runs/<run>/REPORT.md (DESIGN.md §6).

    uv run python -m training.eval.run --ckpt best.pt [--run s-v1] [--config configs/s.yaml] [--scenes 24]
    uv run python -m training.eval.run --ckpt random --config configs/s.yaml --run eval-random

A bare file name (e.g. best.pt) is looked up in runs/<run>/. Must finish in <= 8 min on the VM.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time

import numpy as np
import torch
import yaml

from training.data import loop_sim
from training.eval import asg as A
from training.eval import metrics as M
from training.export.rtneural_json import _git_sha
from training.model.net import RinglessNet
from training.model.streaming import Streaming

ROOT = pathlib.Path(__file__).resolve().parents[1]
THRESH = {"asg_median": 3.0, "asg_p10": 1.0, "sisdri_m3": 3.0, "dstoi_m6": -0.03, "latency_ms": 4.0, "params_s": 250_000}


def streaming_check(net: RinglessNet) -> float:
    torch.manual_seed(0)
    x = 0.05 * torch.randn(2, 48 * 200)
    with torch.no_grad():
        y = net(x)
    s = Streaming(net); s.reset(2)
    ys = torch.cat([s.step(x[:, i:i + 96]) for i in range(0, x.shape[1], 96)], 1)
    return float((ys[:, 48:] - y[:, :-48]).abs().max())


def _evaluate_part(args) -> dict:
    state, cfg, indices, seconds = args
    torch.set_num_threads(1)
    net = RinglessNet.from_config(cfg); net.load_state_dict(state); net.eval()
    return evaluate(net, cfg, seconds=seconds, indices=indices, workers=1)


def evaluate(net: RinglessNet, cfg: dict, n_scenes: int | None = None, seconds: float | None = None,
             indices=None, workers: int = 4) -> dict:
    """G2c metrics over held-out scenes; with workers > 1 the scenes are split across processes."""
    ev = cfg["eval"]
    t0 = time.time()
    if indices is None:
        indices = list(range(n_scenes or ev["scenes"]))
    if workers > 1 and len(indices) > 1:
        import multiprocessing as mp
        parts = [indices[k::workers] for k in range(workers) if indices[k::workers]]
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        with mp.get_context("spawn").Pool(len(parts)) as pool:
            outs = pool.map(_evaluate_part, [(state, cfg, p, seconds) for p in parts])
        order = np.argsort(np.concatenate(parts))
        res = {}
        for k in ("msg_identity", "msg_proc", "asg", "flags_identity", "flags_proc", "sisdri", "dstoi",
                  "sisdr_id_m3_all", "sisdr_md_m3_all", "howl_id_all", "howl_md_all", "level_all"):
            res[k] = np.concatenate([o[k] for o in outs])[order]
        res.update(n_scenes=len(indices), seconds=outs[0]["seconds"])
        return _summarise(res, t0)
    sc = A.Scenes(cfg, seconds=seconds, indices=indices)
    lo, hi = ev["search_db"]
    res = A.asg(sc, lambda: Streaming(net), loop_sim.Identity, lo, hi, ev["search_iters"])
    skip = FS_SKIP = 24000                                                    # first 0.5 s excluded
    D = 48                                                                    # stream delay
    out = {}
    for g, key in ((-3.0, "m3"), (-6.0, "m6"), (0.0, "0"), (-12.0, "m12")):
        r_id = sc.run(loop_sim.Identity(), np.full(sc.n, g))
        r_md = sc.run(Streaming(net), np.full(sc.n, g))
        out[key] = (r_id.mic.numpy(), r_md.p.numpy())
    tgt = sc.target
    si_id = [M.sisdr(out["m3"][0][i, skip:], tgt[i, skip:]) for i in range(sc.n)]
    si_md = [M.sisdr(out["m3"][1][i, skip + D:], tgt[i, skip:-D]) for i in range(sc.n)]
    st_id = [M.stoi(out["m6"][0][i, skip:], tgt[i, skip:]) for i in range(sc.n)]
    st_md = [M.stoi(out["m6"][1][i, skip + D:], tgt[i, skip:-D]) for i in range(sc.n)]
    he_id = [M.howl_energy(out["0"][0][i, skip:], tgt[i, skip:]) for i in range(sc.n)]
    he_md = [M.howl_energy(out["0"][1][i, skip + D:], tgt[i, skip:-D]) for i in range(sc.n)]
    lvl = [M.level_offset_db(out["m12"][1][i, skip + D:], tgt[i, skip:-D]) for i in range(sc.n)]
    res.update(sisdri=np.array(si_md) - np.array(si_id), dstoi=np.array(st_md) - np.array(st_id),
               sisdr_id_m3_all=np.array(si_id), sisdr_md_m3_all=np.array(si_md), howl_id_all=np.array(he_id),
               howl_md_all=np.array(he_md), level_all=np.array(lvl), n_scenes=sc.n, seconds=sc.T / 48000)
    return _summarise(res, t0)


def _summarise(res: dict, t0: float) -> dict:
    res.update(asg_median=float(np.median(res["asg"])), asg_p10=float(np.percentile(res["asg"], 10)),
               sisdri_median=float(np.median(res["sisdri"])), dstoi_median=float(np.median(res["dstoi"])),
               sisdr_id_m3=float(np.median(res["sisdr_id_m3_all"])), sisdr_model_m3=float(np.median(res["sisdr_md_m3_all"])),
               howl_ratio_id_0=float(np.median(res["howl_id_all"])), howl_ratio_model_0=float(np.median(res["howl_md_all"])),
               level_offset_m12=float(np.median(res["level_all"])), wall_s=round(time.time() - t0, 1))
    return res


def gate(res: dict, net: RinglessNet, stream_err: float) -> tuple[bool, list]:
    lat = net.latency_samples / 48.0
    rows = [("ASG median", f"{res['asg_median']:+.2f} dB", "≥ 3 dB", res["asg_median"] >= THRESH["asg_median"]),
            ("ASG 10th percentile", f"{res['asg_p10']:+.2f} dB", "≥ 1 dB", res["asg_p10"] >= THRESH["asg_p10"]),
            ("SI-SDR improvement at −3 dB (median)", f"{res['sisdri_median']:+.2f} dB", "≥ 3 dB", res["sisdri_median"] >= THRESH["sisdri_m3"]),
            ("STOI change at −6 dB (median)", f"{res['dstoi_median']:+.3f}", "≥ −0.03", res["dstoi_median"] >= THRESH["dstoi_m6"]),
            ("Streaming = full sequence", f"max err {stream_err:.1e}", "≤ 1e-4 (test green)", stream_err <= 1e-4),
            ("Algorithmic latency", f"{lat:.1f} ms", "≤ 4 ms", lat <= THRESH["latency_ms"]),
            ("Parameters", f"{net.param_count():,}", "≤ 250k", net.param_count() <= THRESH["params_s"])]
    return all(r[3] for r in rows), rows


def write_report(path: pathlib.Path, res: dict, net, cfg, ckpt_desc: str, step: int, phase, config: str, notes: str = "") -> bool:
    stream_err = streaming_check(net)
    ok, rows = gate(res, net, stream_err)
    tpl = (ROOT / "eval" / "REPORT_TEMPLATE.md").read_text()
    scene_rows = "\n".join(f"| {i} | {res['msg_identity'][i]:+.2f} | {res['msg_proc'][i]:+.2f} | {res['asg'][i]:+.2f} | "
                           f"{(res['flags_identity'][i] + ' ' + res['flags_proc'][i]).strip()} |" for i in range(res["n_scenes"]))
    extra = [("Median SI-SDR at −3 dB, identity / model", f"{res['sisdr_id_m3']:.2f} / {res['sisdr_model_m3']:.2f} dB"),
             ("Howl-energy ratio at 0 dB, identity / model", f"{res['howl_ratio_id_0']:.3f} / {res['howl_ratio_model_0']:.3f}"),
             ("Level offset at −12 dB (model output vs target)", f"{res['level_offset_m12']:+.2f} dB"),
             ("Median MSG identity / model (re estimate)", f"{np.median(res['msg_identity']):+.2f} / {np.median(res['msg_proc']):+.2f} dB")]
    ev = cfg["eval"]
    text = tpl.format(run=cfg.get("run", ""), ckpt=ckpt_desc, step=step, phase=phase, config=config, git_sha=_git_sha()[:12],
                      created=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), n_scenes=res["n_scenes"],
                      seconds=res["seconds"], search_lo=ev["search_db"][0], search_hi=ev["search_db"][1], iters=ev["search_iters"],
                      wall_s=res["wall_s"], verdict="PASS" if ok else "FAIL",
                      gate_rows="\n".join(f"| {a} | {b} | {c} | {'PASS' if d else 'FAIL'} |" for a, b, c, d in rows),
                      scene_rows=scene_rows, extra_rows="\n".join(f"| {a} | {b} |" for a, b in extra),
                      fsr_max=ev["fsr_max_db"], notes=notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    js.update(verdict="PASS" if ok else "FAIL", stream_err=stream_err, step=step, ckpt=ckpt_desc)
    path.with_name("report.json").write_text(json.dumps(js, indent=1))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path, a file name inside runs/<run>/, or 'random'")
    ap.add_argument("--run", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="processes; scenes are split across them")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import os
    run = a.run or os.environ.get("RL_RUN", "s-v1")
    if a.ckpt == "random":
        cfg = yaml.safe_load(open(a.config or ROOT / "configs" / "s.yaml"))
        torch.manual_seed(cfg["seed"])
        net, step, phase, config = RinglessNet.from_config(cfg), 0, "-", a.config or "configs/s.yaml"
    else:
        p = pathlib.Path(a.ckpt)
        if not p.exists():
            p = ROOT / "runs" / run / a.ckpt
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["config"] if not a.config else yaml.safe_load(open(a.config))
        net = RinglessNet.from_config(cfg); net.load_state_dict(ck["model"])
        step, phase, config = ck["step"], ck["phase"], ck.get("config_path", "")
    net.eval()
    cfg = dict(cfg); cfg["run"] = run
    res = evaluate(net, cfg, a.scenes, a.seconds, workers=a.workers)
    out = ROOT / "runs" / run / "REPORT.md"
    ok = write_report(out, res, net, cfg, a.ckpt, step, phase, config)
    print(f"G2c: {'PASS' if ok else 'FAIL'} — ASG median {res['asg_median']:+.2f} dB, p10 {res['asg_p10']:+.2f} dB, "
          f"SI-SDRi {res['sisdri_median']:+.2f} dB, dSTOI {res['dstoi_median']:+.3f} ({res['wall_s']} s) -> {out}")


if __name__ == "__main__":
    main()
