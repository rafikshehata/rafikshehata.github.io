"""NumPy re-implementation of the runtime: docs/model-format.md §1 framing + RTNeural layer math.

Reads only model.json and model.rtneural.json. GRU gate order z|r|h and the
two-bias reset-after form follow RTNeural commit 95c3c0f (RTNeural/gru/gru.h:
z = σ(Wz x + Uz h + b0z + b1z), r = σ(...), c = tanh(Wc x + b0c + r ⊙ (Uc h + b1c)),
h' = (1 - z) ⊙ c + z ⊙ h). Dense: y = Kᵀ x + b (RTNeural model_loader.h loadDense).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RTNeuralModel:
    """Minimal dynamic-model equivalent of RTNeural::Model for gru/dense(+activation) layers (float64)."""

    def __init__(self, j: dict):
        self.layers = []
        for l in j["layers"]:
            if l["type"] == "gru":
                W, U, b = (np.asarray(w, np.float64) for w in l["weights"])
                h = W.shape[1] // 3
                self.layers.append(("gru", W, U, b, h))
            elif l["type"] == "dense":
                K, b = (np.asarray(w, np.float64) for w in l["weights"])
                self.layers.append(("dense", K, b, l.get("activation", "")))
            else:
                raise ValueError(f"unsupported layer {l['type']}")
        self.reset()

    def reset(self):
        self.state = [np.zeros(l[4]) if l[0] == "gru" else None for l in self.layers]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, l in enumerate(self.layers):
            if l[0] == "gru":
                _, W, U, b, h = l
                hp = self.state[i]
                xw = x @ W + b[0]                      # input part, columns z | r | c
                hu = hp @ U + b[1]                     # recurrent part
                z = _sigmoid(xw[:h] + hu[:h])
                r = _sigmoid(xw[h:2 * h] + hu[h:2 * h])
                c = np.tanh(xw[2 * h:] + r * hu[2 * h:])
                x = (1 - z) * c + z * hp
                self.state[i] = x
            else:
                _, K, b, act = l
                x = x @ K + b
                if act == "sigmoid":
                    x = _sigmoid(x)
                elif act == "tanh":
                    x = np.tanh(x)
                elif act == "relu":
                    x = np.maximum(x, 0)
                elif act not in ("", "linear"):
                    raise ValueError(act)
        return x


class ReferenceStage:
    """The full neural stage (pre, RTNeural, post, OLA) in NumPy, block = hop samples."""

    def __init__(self, export_dir):
        d = pathlib.Path(export_dir)
        self.side = json.loads((d / "model.json").read_text())
        self.net = RTNeuralModel(json.loads((d / self.side["network"]["file"]).read_text()))
        self.W, self.H = self.side["window"], self.side["hop"]
        self.A = np.asarray(self.side["analysis"]["matrix"], np.float64)
        self.S = np.asarray(self.side["synthesis"]["matrix"], np.float64)
        self.rows = self.side["bins"]["rows"]
        f = self.side["features"]
        self.eps, self.offset, self.scale = f["eps"], f["offset"], f["scale"]
        self.reset()

    def reset(self):
        self.x = np.zeros(self.W); self.acc = np.zeros(self.W); self.net.reset()
        self.masks = []

    def process_block(self, u: np.ndarray) -> np.ndarray:
        H = self.H
        self.x = np.concatenate([self.x[H:], u])
        c = self.A @ self.x
        P = np.array([sum(c[r] ** 2 for r in rr) for rr in self.rows])
        f = self.scale * (np.log10(P + self.eps) + self.offset)
        m = self.net.forward(f)
        self.masks.append(m)
        g = np.zeros(self.W)
        for k, rr in enumerate(self.rows):
            g[rr] = m[k] * c[rr]
        self.acc += self.S @ g
        out = self.acc[:H].copy()
        self.acc = np.concatenate([self.acc[H:], np.zeros(H)])
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        self.reset()
        return np.concatenate([self.process_block(x[i:i + self.H]) for i in range(0, len(x), self.H)])
