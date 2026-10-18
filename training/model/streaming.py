"""Frame-by-frame inference with carried state (DESIGN.md §2, docs/model-format.md §1).

`Streaming.step(block)` consumes any multiple of H samples and returns as many;
output sample m equals the full-sequence estimate at m - 48 (stream delay W - H).
Also usable as a loop_sim processor (reset/step interface).
"""
from __future__ import annotations

import torch

from training.model import fb
from training.model.net import RinglessNet


class Streaming:
    def __init__(self, net: RinglessNet):
        self.net = net
        self.W, self.H = net.window, net.hop
        self.reset(1)

    def reset(self, batch: int):
        self.x = torch.zeros(batch, self.W)
        self.acc = torch.zeros(batch, self.W)
        self.h1 = self.h2 = None

    @torch.no_grad()
    def step(self, block: torch.Tensor) -> torch.Tensor:
        net, W, H = self.net, self.W, self.H
        outs = []
        for i in range(0, block.shape[-1], H):
            self.x = torch.cat([self.x[:, H:], block[:, i:i + H]], 1)
            c = self.x @ net.A.T                                          # (B, 96)
            m, self.h1, self.h2 = net.mask(net.features(c).unsqueeze(1), self.h1, self.h2)
            z = (fb.expand_mask(m[:, 0]) * c) @ net.S.T
            self.acc = self.acc + z
            outs.append(self.acc[:, :H])
            self.acc = torch.cat([self.acc[:, H:], torch.zeros_like(self.acc[:, :H])], 1)
        return torch.cat(outs, 1)

    @property
    def stream_delay(self) -> int:
        return self.W - self.H
