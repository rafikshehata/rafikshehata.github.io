"""Export a RinglessNet to RTNeural JSON + model.json sidecar (docs/model-format.md).

    uv run python -m training.export.rtneural_json --ckpt runs/s-v1/best.pt --out runs/s-v1/export
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess

import numpy as np
import torch

from training.model import fb
from training.model.net import RinglessNet

RTNEURAL_COMMIT = "95c3c0f987a6fe903e7eec71e797405dbed7caf7"
FORMAT = "ringless-neural-v1"


def _gru_layer(gru: torch.nn.GRU) -> dict:
    """PyTorch r|z|n row blocks -> RTNeural/Keras z|r|h columns (kernel in×3h, recurrent h×3h, bias 2×3h)."""
    h = gru.hidden_size
    def reorder(w):                                   # (3h, ...) rows r, z, n -> z, r, n
        r, z, n = w[:h], w[h:2 * h], w[2 * h:]
        return torch.cat([z, r, n], 0)
    W = reorder(gru.weight_ih_l0.detach()).T.contiguous()
    U = reorder(gru.weight_hh_l0.detach()).T.contiguous()
    b = torch.stack([reorder(gru.bias_ih_l0.detach()), reorder(gru.bias_hh_l0.detach())])
    return {"type": "gru", "activation": "tanh", "shape": [None, None, h],
            "weights": [W.tolist(), U.tolist(), b.tolist()]}


def _dense_layer(dense: torch.nn.Linear, activation: str) -> dict:
    return {"type": "dense", "activation": activation, "shape": [None, None, dense.out_features],
            "weights": [dense.weight.detach().T.tolist(), dense.bias.detach().tolist()]}


def rtneural_dict(net: RinglessNet) -> dict:
    return {"in_shape": [None, None, net.n_bins],
            "layers": [_gru_layer(net.gru1), _gru_layer(net.gru2), _dense_layer(net.dense, "sigmoid")]}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).parent).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def export(net: RinglessNet, out_dir, name: str = "s-v1", run: str = "", step: int = 0,
           config: str = "", gate: dict | None = None) -> pathlib.Path:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rt_bytes = json.dumps(rtneural_dict(net)).encode()
    (out / "model.rtneural.json").write_bytes(rt_bytes)
    side = {
        "format": FORMAT, "name": name, "sample_rate": 48000,
        "window": net.window, "hop": net.hop,
        "latency_samples": net.window + net.hop, "latency_ms": (net.window + net.hop) / 48.0,
        "stream_delay": net.window - net.hop,
        "analysis": {"matrix": net.A.detach().tolist()},
        "synthesis": {"matrix": net.S.detach().tolist()},
        "bins": {"count": net.n_bins, "rows": fb.bin_rows(net.window)},
        "features": {"type": "log10_pair_power", "eps": net.feat_eps, "offset": net.feat_offset, "scale": net.feat_scale},
        "network": {"file": "model.rtneural.json", "sha256": hashlib.sha256(rt_bytes).hexdigest(),
                    "input_size": net.n_bins, "output_size": net.n_bins,
                    "layers": [{"type": "gru", "size": net.hidden[0]}, {"type": "gru", "size": net.hidden[1]},
                               {"type": "dense", "size": net.n_bins, "activation": "sigmoid"}],
                    "rtneural_commit": RTNEURAL_COMMIT},
        "params": {"total": int(sum(p.numel() for p in [net.A, net.S]) + net.network_param_count()),
                   "network": int(net.network_param_count())},
        "provenance": {"run": run, "step": int(step), "git_sha": _git_sha(), "config": config,
                       "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
    }
    if gate:
        side["gate"] = gate
    (out / "model.json").write_text(json.dumps(side))
    validate_sidecar(out / "model.json")
    return out


REQUIRED = {
    "format": str, "name": str, "sample_rate": int, "window": int, "hop": int, "latency_samples": int,
    "latency_ms": (int, float), "stream_delay": int, "analysis.matrix": list, "synthesis.matrix": list,
    "bins.count": int, "bins.rows": list, "features.type": str, "features.eps": (int, float),
    "features.offset": (int, float), "features.scale": (int, float), "network.file": str, "network.sha256": str,
    "network.input_size": int, "network.output_size": int, "network.layers": list, "network.rtneural_commit": str,
    "params.total": int, "params.network": int, "provenance.run": str, "provenance.step": int,
    "provenance.git_sha": str, "provenance.config": str, "provenance.created": str,
}


def validate_sidecar(path) -> dict:
    """Check model.json against docs/model-format.md §3; raises ValueError on any mismatch."""
    path = pathlib.Path(path)
    side = json.loads(path.read_text())
    def get(key):
        v = side
        for part in key.split("."):
            if not isinstance(v, dict) or part not in v:
                raise ValueError(f"missing key {key}")
            v = v[part]
        return v
    for key, typ in REQUIRED.items():
        v = get(key)
        if isinstance(v, bool) or not isinstance(v, typ):
            raise ValueError(f"{key}: wrong type {type(v).__name__}")
    Wn, Hn = side["window"], side["hop"]
    if side["format"] != FORMAT or side["sample_rate"] != 48000:
        raise ValueError("format / sample_rate")
    if side["latency_samples"] != Wn + Hn or side["stream_delay"] != Wn - Hn or side["latency_samples"] > 192:
        raise ValueError("latency fields inconsistent or > 4 ms")
    for k in ("analysis", "synthesis"):
        m = np.asarray(side[k]["matrix"])
        if m.shape != (Wn, Wn) or not np.isfinite(m).all():
            raise ValueError(f"{k}.matrix shape {m.shape}")
    rows = side["bins"]["rows"]
    if len(rows) != side["bins"]["count"] or sorted(r for rr in rows for r in rr) != list(range(Wn)):
        raise ValueError("bins.rows must partition 0..W-1")
    rt = path.parent / side["network"]["file"]
    if hashlib.sha256(rt.read_bytes()).hexdigest() != side["network"]["sha256"]:
        raise ValueError("network.sha256 mismatch")
    j = json.loads(rt.read_text())
    if j["in_shape"][-1] != side["network"]["input_size"] or j["layers"][-1]["shape"][-1] != side["network"]["output_size"]:
        raise ValueError("network sizes mismatch")
    return side


def load_net(ckpt_path) -> tuple[RinglessNet, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = RinglessNet.from_config(ck["config"])
    net.load_state_dict(ck["model"])
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net, ck = load_net(a.ckpt)
    out = export(net, a.out, name=ck["config"].get("run", "s-v1"), run=ck["config"].get("run", ""),
                 step=ck.get("step", 0), config=ck.get("config_path", ""))
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
