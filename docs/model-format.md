# Ringless neural model format (`ringless-neural-v1`)

This is the contract between the Python training track (`training/`, C27) and
the C++ runtime (`NeuralStage`, C30). An export is a directory with exactly two
files:

```
model.rtneural.json   the network RTNeural runs (GRU → GRU → Dense + sigmoid)
model.json            the sidecar: framing, filterbank, features, provenance
```

The RTNeural JSON is the only thing inside the audio callback's neural engine
(D20). Everything else in this document is plain C++ arithmetic that the
runtime implements itself.

## 1. Frame procedure (normative)

Constants come from `model.json`: `W` = `window` (96), `H` = `hop` (48),
`A` = `analysis.matrix` (W×W, row-major), `S` = `synthesis.matrix` (W×W,
row-major), `B` = `bins.count` (49), `pairs` = `bins.rows`.

State: input ring `x[W]` (oldest first, zero-initialised), output accumulator
`acc[W]` (zero-initialised), RTNeural model state (reset with `model.reset()`).

For every block of `H` new input samples `u[0..H-1]`:

1. Shift `x` left by `H` and append `u` → `x` holds the newest `W` samples.
2. Analysis: `c = A · x` (W outputs). The analysis window is already folded into `A`.
3. Features, for each bin `k` in `0..B-1`:
   `P_k = Σ_{r in pairs[k]} c[r]²`,
   `f_k = features.scale · (log10(P_k + features.eps) + features.offset)`.
4. Network: `m = RTNeural.forward(f)` — `B` values in (0, 1).
5. Expand: for each bin `k` and each row `r` in `pairs[k]`: `g[r] = m_k · c[r]`.
6. Synthesis: `z = S · g` (W outputs). The synthesis window is folded into `S`.
7. Overlap-add: `acc[i] += z[i]` for `i` in `0..W-1`.
8. Emit `acc[0..H-1]` as the `H` output samples, shift `acc` left by `H`, zero its last `H`.

With the initial (untrained) filterbank and `m ≡ 1` the output equals the
input delayed by `stream_delay` samples (48). Algorithmic latency, counted as
window + hop, is `latency_samples` = 144 (3.0 ms at 48 kHz, D17 ≤ 4 ms).

The model is trained and exported for **48 kHz only**. A host running at
another rate must resample around the stage.

## 2. `model.rtneural.json`

Standard RTNeural `RTNeural::json_parser::parseJson` input:

```json
{
  "in_shape": [null, null, 49],
  "layers": [
    {"type": "gru",   "activation": "tanh", "shape": [null, null, 128], "weights": [K1, R1, B1]},
    {"type": "gru",   "activation": "tanh", "shape": [null, null, 128], "weights": [K2, R2, B2]},
    {"type": "dense", "activation": "sigmoid", "shape": [null, null, 49], "weights": [Kd, bd]}
  ]
}
```

| Tensor | Shape | Meaning |
|---|---|---|
| `K` (GRU kernel) | `in × 3h` | columns ordered **z \| r \| h** (update, reset, candidate) |
| `R` (GRU recurrent) | `h × 3h` | same column order |
| `B` (GRU bias) | `2 × 3h` | row 0 = input bias, row 1 = recurrent bias, same column order |
| `Kd` (Dense kernel) | `in × out` | `y = Kdᵀ·x + bd` |
| `bd` | `out` | bias |

GRU math (as implemented by RTNeural `GRULayer`, reset-after form):

```
z = σ(Wz·x + Uz·h + bz0 + bz1)
r = σ(Wr·x + Ur·h + br0 + br1)
n = tanh(Wh·x + bh0 + r ⊙ (Uh·h + bh1))
h' = (1 − z) ⊙ n + z ⊙ h
```

This equals PyTorch `nn.GRU` with PyTorch's `r | z | n` row blocks reordered
to `z | r | h` and transposed. The "S" export has hidden sizes 128/128; "M"
has 256/256. The `shape` entries above are what the loader reads; sizes must
match `model.json` → `network`.

RTNeural reference: repository `jatinchowdhury18/RTNeural`, commit
`95c3c0f987a6fe903e7eec71e797405dbed7caf7` (the repository has no release
tags; see `training/DESIGN.md` §9 D5). Files checked: `RTNeural/gru/gru.h`,
`RTNeural/gru/gru.tpp`, `RTNeural/model_loader.h`.

## 3. `model.json` (sidecar)

All keys are required unless marked optional. Matrices are lists of rows.

| Key | Type | Value / meaning |
|---|---|---|
| `format` | string | `"ringless-neural-v1"` |
| `name` | string | model id, e.g. `"s-v1"` |
| `sample_rate` | int | `48000` |
| `window` | int | `96` |
| `hop` | int | `48` |
| `latency_samples` | int | `window + hop` = `144` |
| `latency_ms` | number | `3.0` |
| `stream_delay` | int | `window − hop` = `48`; delay of the identity reconstruction |
| `analysis.matrix` | `W × W` numbers | `A`, analysis window folded in |
| `synthesis.matrix` | `W × W` numbers | `S`, synthesis window folded in |
| `bins.count` | int | `49` |
| `bins.rows` | list of `B` lists | rows of `c` belonging to each bin: `[[0], [1, 49], [2, 50], …, [47, 95], [48]]` |
| `features.type` | string | `"log10_pair_power"` |
| `features.eps` | number | `1e-10` |
| `features.offset` | number | `5.0` |
| `features.scale` | number | `0.5` |
| `network.file` | string | `"model.rtneural.json"` |
| `network.sha256` | string | hex SHA-256 of that file's bytes |
| `network.input_size` | int | `49` |
| `network.output_size` | int | `49` |
| `network.layers` | list | e.g. `[{"type": "gru", "size": 128}, {"type": "gru", "size": 128}, {"type": "dense", "size": 49, "activation": "sigmoid"}]` |
| `network.rtneural_commit` | string | pinned RTNeural commit (above) |
| `params.total` | int | trainable parameters incl. filterbank |
| `params.network` | int | parameters inside the RTNeural file |
| `provenance.run` | string | training run id |
| `provenance.step` | int | training step of the exported checkpoint |
| `provenance.git_sha` | string | commit of `training/` that produced it |
| `provenance.config` | string | config file name |
| `provenance.created` | string | ISO-8601 UTC timestamp |
| `gate` | object, optional | G2c numbers from `REPORT.md` when available |

## 4. Validation

`training/export/rtneural_json.py` writes both files and validates the sidecar
against this table (`validate_sidecar`). `training/export/reference_infer.py`
re-implements §1 and §2 in NumPy straight from the two files; the export test
requires it to match PyTorch within 1e−4.
