"""Parameter count for the DESIGN.md architectures (Planner script, not pipeline code).

Usage: uv run python scripts/param_count.py
Counts torch layers equivalent to DESIGN.md section 3: learnable 96x96 analysis
and synthesis matrices, two GRUs, one Dense mask layer with 49 outputs.
"""
import torch

W, N_BINS = 96, 49
ARCH = {"tiny": ([16, 16], False), "s": ([128, 128], True), "m": ([256, 256], True)}

for name, (hidden, learn_fb) in ARCH.items():
    layers = [torch.nn.GRU(N_BINS, hidden[0], batch_first=True),
              torch.nn.GRU(hidden[0], hidden[1], batch_first=True),
              torch.nn.Linear(hidden[1], N_BINS)]
    body = sum(p.numel() for l in layers for p in l.parameters())
    fb = 2 * W * W if learn_fb else 0
    per = [sum(p.numel() for p in l.parameters()) for l in layers]
    print(f"{name}: gru1={per[0]} gru2={per[1]} dense={per[2]} filterbank={fb} "
          f"trainable_total={body + fb}")
