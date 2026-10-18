"""B3: smoke test of the chunked, resumable training loop with configs/tiny.yaml (< 3 min)."""
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(run, *extra, check=True):
    cmd = [PY, "train.py", "--config", "configs/tiny.yaml", "--run", run, "--max-minutes", "0", *extra]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=170)
    if check:
        assert p.returncode == 0, p.stdout + p.stderr
    return p


@pytest.fixture(scope="module")
def runs():
    names = ["_smoke_a", "_smoke_b", "_smoke_c"]
    for n in names:
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
    yield {n: ROOT / "runs" / n for n in names}
    for n in names:
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


def _state(d):
    return json.loads((d / "state.json").read_text())


def test_kill_and_resume_is_bit_exact(runs):
    t0 = time.time()
    a, b = runs["_smoke_a"], runs["_smoke_b"]
    _run("_smoke_a", "--stop-at-step", "50")                         # 50 steps, then stop + checkpoint
    assert _state(a)["step"] == 50
    ck = torch.load(a / "latest.pt", weights_only=False)
    assert ck["step"] == 50 and len(ck["opt"]["state"]) > 0
    p = _run("_smoke_a", "--resume")                                  # continues at 51 .. 60
    assert "resumed at step 50" in p.stdout and "adam_step=" in p.stdout
    lines = [json.loads(l) for l in open(a / "metrics.jsonl")]
    steps = [r["step"] for r in lines if "loss" in r]
    assert steps == sorted(steps) and steps[-1] == 60 and 60 in steps
    assert _state(a)["step"] == 60 and _state(a)["done"]
    _run("_smoke_b")                                                  # straight 60 steps
    wa = torch.load(a / "latest.pt", weights_only=False)["model"]
    wb = torch.load(b / "latest.pt", weights_only=False)["model"]
    for k in wa:
        assert torch.equal(wa[k], wb[k]), f"resume not bit-exact for {k}"
    for f in ("latest.pt", "prev.pt", "best.pt", "state.json", "metrics.jsonl"):
        assert (a / f).exists(), f
    assert time.time() - t0 < 170


def test_sigterm_checkpoints_and_exits_zero(runs):
    c = runs["_smoke_c"]
    proc = subprocess.Popen([PY, "train.py", "--config", "configs/tiny.yaml", "--run", "_smoke_c", "--max-minutes", "0"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        if (c / "metrics.jsonl").exists() and any("loss" in l for l in open(c / "metrics.jsonl")):
            break
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    st = _state(c)
    ck = torch.load(c / "latest.pt", weights_only=False)
    assert ck["step"] == st["step"] > 0 and st["reason"] == "sigterm/stop"


def test_metrics_keys(runs):
    lines = [json.loads(l) for l in open(runs["_smoke_a"] / "metrics.jsonl")]
    train = [r for r in lines if "loss" in r]; val = [r for r in lines if "val_sisdr" in r]
    assert train and val
    for k in ("step", "phase", "loss", "l_mr", "sisdr", "l_howl", "lr", "steps_per_s", "elapsed"):
        assert k in train[0]
    for k in ("step", "val_loss", "val_sisdr", "best_val_sisdr"):
        assert k in val[0]
    assert {r["phase"] for r in train} == {1, 2}


def test_two_rank_resume_is_bit_exact():
    names = ["_smoke_d", "_smoke_e"]
    for n in names:
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
    try:
        _run("_smoke_d", "--procs", "2", "--stop-at-step", "20")
        assert _state(ROOT / "runs" / "_smoke_d")["step"] == 20
        p = _run("_smoke_d", "--procs", "2", "--resume", "--max-steps", "30")
        assert "resumed at step 20" in p.stdout and _state(ROOT / "runs" / "_smoke_d")["step"] == 30
        _run("_smoke_e", "--procs", "2", "--max-steps", "30")
        wa = torch.load(ROOT / "runs" / "_smoke_d" / "latest.pt", weights_only=False)["model"]
        wb = torch.load(ROOT / "runs" / "_smoke_e" / "latest.pt", weights_only=False)["model"]
        assert all(torch.equal(wa[k], wb[k]) for k in wa)
    finally:
        for n in names:
            shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
