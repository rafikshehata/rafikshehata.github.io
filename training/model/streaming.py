"""Frame-by-frame inference with carried state (DESIGN.md §2, docs/model-format.md §1).

`Streaming.step(block)` consumes any multiple of H samples and returns as many;
output sample m equals the full-sequence estimate at m - 48 (stream delay W - H).
Also usable as a loop_sim processor (reset/step interface).

The per-hop path is written for speed (it runs 1 000 times per simulated
second in phase 2 and eval): fused GRU cells, bin pairing and mask expansion as
0/1 matrix products, and a half-frame overlap-add tail (W = 2H).
"""
from __future__ import annotations

import torch

from training.model import fb
from training.model.net import RinglessNet


class Streaming:
    def __init__(self, net: RinglessNet):
        self.net = net
        self.W, self.H = net.window, net.hop
        assert self.W == 2 * self.H
        rows = fb.bin_rows(self.W)
        pair = torch.zeros(self.W, len(rows))
        for k, rr in enumerate(rows):
            pair[rr, k] = 1.0
        self.pair, self.expand = pair, pair.T.contiguous()
        self.reset(1)

    def reset(self, batch: int):
        self.x = torch.zeros(batch, self.H)            # previous hop of input
        self.tail = torch.zeros(batch, self.H)         # second half of the previous output frame
        self.h1 = torch.zeros(batch, self.net.hidden[0])
        self.h2 = torch.zeros(batch, self.net.hidden[1])

    @torch.no_grad()
    def step(self, block: torch.Tensor) -> torch.Tensor:
        net, H = self.net, self.H
        g1, g2, d = net.gru1, net.gru2, net.dense
        At, St = net.A.T, net.S.T
        outs = []
        for i in range(0, block.shape[-1], H):
            u = block[:, i:i + H]
            c = torch.cat([self.x, u], 1) @ At                                   # (B, 96)
            self.x = u
            f = net.feat_scale * (torch.log10((c * c) @ self.pair + net.feat_eps) + net.feat_offset)
            self.h1 = torch._VF.gru_cell(f, self.h1, g1.weight_ih_l0, g1.weight_hh_l0, g1.bias_ih_l0, g1.bias_hh_l0)
            self.h2 = torch._VF.gru_cell(self.h1, self.h2, g2.weight_ih_l0, g2.weight_hh_l0, g2.bias_ih_l0, g2.bias_hh_l0)
            m = torch.sigmoid(torch.addmm(d.bias, self.h2, d.weight.T))
            z = ((m @ self.expand) * c) @ St
            outs.append(self.tail + z[:, :H])
            self.tail = z[:, H:]
        return torch.cat(outs, 1)

    @property
    def stream_delay(self) -> int:
        return self.W - self.H
