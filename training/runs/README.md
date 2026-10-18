# Training runs

Everything in this directory except this file is gitignored. A run's files live
in `runs/<run>/` on the VM and are published to the orphan branch `ckpt/<run>`
(one force-pushed commit) by `scripts/ckpt_push.sh` after every chunk.

## Files of a run

| File | Content |
|---|---|
| `latest.pt` | newest checkpoint: model, optimizer, step, phase, best metric, config, RNG state |
| `prev.pt` | the checkpoint `latest.pt` replaced (one step of undo) |
| `best.pt` | checkpoint with the best validation SI-SDR so far |
| `state.json` | one-glance summary: step, phase, best, last validation, chunks, train seconds, NaN steps |
| `metrics.jsonl` | one JSON object per line, see below |
| `chunks.jsonl` | one line per chunk from `chunk.sh`: start/end time, minutes, exit code, log file |
| `REPORT.md`, `report.json` | gate G2c evaluation from `python -m training.eval.run` |
| `chunk-*.log` | stdout of the last three chunks (not pushed) |

## Reading `metrics.jsonl`

Two kinds of lines, both with `step` and `phase`:

* **Training** (every `log_every` steps): `loss` (total), `l_mr` (multi-resolution
  STFT), `sisdr` (training-batch SI-SDR, dB), `l_howl` (howl penalty), `lr`,
  `grad_norm`, `steps_per_s`, `elapsed` (training seconds), and `id_frac`
  (phase 1, share of identity-loop pairs) or `trip_rate` (phase 2, share of
  rollouts whose saturation guard tripped).
* **Validation** (every `val_every` steps and at each exit): `val_loss`,
  `val_sisdr` (mean over the 200 fixed held-out pairs), `val_sisdr_in` (the
  unprocessed input, for reference), `val_sisdr_tf` / `val_sisdr_id` (split by
  pair type), `best_val_sisdr`, `best_step`.

Quick look:

```sh
grep val_sisdr runs/s-v1/metrics.jsonl | tail -n 5
bash scripts/status.sh s-v1
```

## Resuming

In the cloud, one chunk at a time: `bash scripts/chunk.sh s-v1 25` (pulls
`ckpt/s-v1`, trains up to 25 minutes, pushes). Locally (M1 Pro), restore and
run without a time limit:

```sh
bash scripts/ckpt_pull.sh s-v1
uv run python train.py --config configs/s.yaml --run s-v1 --resume --max-minutes 0
```

Resuming is bit-exact for the same number of ranks (`train.procs`): the data of
step s depends only on the run seed and s.
