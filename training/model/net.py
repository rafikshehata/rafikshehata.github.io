"""tiny / S / M networks (DESIGN.md §3): C++-side filterbank + RTNeural-side GRU-GRU-Dense-sigmoid."""
from __future__ import annotations

import torch
import torch.nn as nn

from training.model import fb


class RinglessNet(nn.Module):
    def __init__(self, hidden=(128, 128), learn_filterbank: bool = True, mask_bias_init: float = 3.0,
                 feat_eps: float = 1e-10, feat_offset: float = 5.0, feat_scale: float = 0.5,
                 window: int = fb.W, hop: int = fb.H):
        super().__init__()
        assert (window, hop) == (fb.W, fb.H), "s-v1 front-end is fixed at W=96, H=48"
        self.window, self.hop, self.n_bins = window, hop, window // 2 + 1
        self.feat_eps, self.feat_offset, self.feat_scale = feat_eps, feat_offset, feat_scale
        A = torch.tensor(fb.analysis_init(window), dtype=torch.float32)
        S = torch.tensor(fb.synthesis_init(window), dtype=torch.float32)
        if learn_filterbank:
            self.A, self.S = nn.Parameter(A), nn.Parameter(S)
        else:
            self.register_buffer("A", A); self.register_buffer("S", S)
        self.gru1 = nn.GRU(self.n_bins, hidden[0], batch_first=True)
        self.gru2 = nn.GRU(hidden[0], hidden[1], batch_first=True)
        self.dense = nn.Linear(hidden[1], self.n_bins)
        nn.init.constant_(self.dense.bias, mask_bias_init)
        self.hidden = tuple(hidden)

    @classmethod
    def from_config(cls, cfg: dict) -> "RinglessNet":
        m = cfg["model"]
        return cls(hidden=tuple(m["hidden"]), learn_filterbank=m["learn_filterbank"], mask_bias_init=m["mask_bias_init"],
                   feat_eps=m["feat_eps"], feat_offset=m["feat_offset"], feat_scale=m["feat_scale"],
                   window=cfg["audio"]["window"], hop=cfg["audio"]["hop"])

    @property
    def latency_samples(self) -> int:
        return self.window + self.hop                      # D17 counting: window + one hop

    def features(self, c: torch.Tensor) -> torch.Tensor:
        return self.feat_scale * (torch.log10(fb.pair_power(c) + self.feat_eps) + self.feat_offset)

    def mask(self, f, h1=None, h2=None):
        y1, h1 = self.gru1(f, h1)
        y2, h2 = self.gru2(y1, h2)
        return torch.sigmoid(self.dense(y2)), h1, h2

    def forward(self, x: torch.Tensor, tbptt: int | None = None, return_mask: bool = False):
        """Full-sequence processing: (B, T) mic -> (B, T) estimate aligned with the target.

        tbptt: detach the GRU states every `tbptt` frames (phase-2 truncated BPTT); same forward values.
        """
        T = x.shape[-1]
        c = fb.frames(x, self.window, self.hop) @ self.A.T          # (B, Tf, 96)
        f = self.features(c)
        if tbptt:
            ms, h1, h2 = [], None, None
            for t0 in range(0, f.shape[1], tbptt):
                m, h1, h2 = self.mask(f[:, t0:t0 + tbptt], h1, h2)
                ms.append(m); h1, h2 = h1.detach(), h2.detach()
            m = torch.cat(ms, 1)
        else:
            m, _, _ = self.mask(f)
        z = (fb.expand_mask(m) * c) @ self.S.T
        y = fb.overlap_add(z, T, self.window, self.hop)
        return (y, m) if return_mask else y

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def network_param_count(self) -> int:
        return sum(p.numel() for mod in (self.gru1, self.gru2, self.dense) for p in mod.parameters())
