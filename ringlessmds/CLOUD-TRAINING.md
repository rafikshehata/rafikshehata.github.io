# Ringless — Cloud Training Runbook (neural stage, Stage 3 in Claude Code cloud sessions)

> **Who reads this:** three kinds of Claude Code *cloud* sessions — a **Planner**
> (Opus / Fable), **Builders** (Sonnet) and a **Babysitter** (Haiku) — plus the
> owner (Raf). Every session reads this file first, then only what its role
> section tells it to read. This runbook replaces the local-compute assumption
> of `03-advanced.md` for the training-side chunks (C25 bench, C26 data,
> C27 model/export, C28 training, C29 Python eval). The C++ runtime chunks
> (C30 NeuralStage, C31 hybrid wiring, C32 signing) stay local in `~/Workbench`.
> `00-overview.md` Purpose / Non-Goals / Domain rules and `research.md`
> D17–D21 still bind every agent here.

## 0. Goal and definition of done

**Goal.** Produce, entirely from cloud sessions, a trained and exported
small neural voice-separation / anti-howl model (`s-v1`) for Ringless's
neural stage, plus the pipeline that made it:

1. `training/` — resumable PyTorch pipeline: feedback-loop data simulator,
   model, export to RTNeural JSON, chunked training loop, eval harness, tests.
2. A trained `s-v1` checkpoint on the `ckpt/s-v1` git branch, an exported
   `model.rtneural.json` + `model.json`, and `REPORT.md` with the gate numbers.

**Done (cloud track) when all of these hold:**
- `uv run pytest -q` is green on `main`.
- `train.py --resume --max-minutes 25` runs, checkpoints, exits 0 and resumes
  bit-exactly (step counter and optimizer state).
- Gate **G2c** (§10) is **PASS**, *or* an honest **FAIL** report exists after the
  chunk budget (§9) with the numbers and a diagnosis request for the Planner.
- The STATE block at the bottom of this file reflects reality.

What "done" hands back to the local track (§10): the export for C30, the
full C29 gate (needs the C++ notch engine, C7) and the long M-size run.

## 1. Hard constraints

**Cloud VM (Anthropic-hosted, per session):** Ubuntu 24.04 x86_64, 4 vCPUs,
16 GB RAM, 30 GB disk, **no GPU**. A fresh VM each session; only the
repository clone and whatever a cached setup script wrote survive between
sessions. Numbers are approximate ceilings and may change.

**Time limits (why everything is chunked):** a command Claude runs in the
foreground is cut at 10 minutes; a command moved to the background is killed
after 30 more minutes; an idle VM pauses after a few minutes and can be
reclaimed, losing running processes. Therefore: **no single process may run
longer than 25 minutes**, every long job checkpoints to disk at least every
5 minutes and at exit, and every checkpoint is pushed off the VM (§8) before
the session ends its turn.

**Disk budget (30 GB total, keep ≥ 3 GB free):**

| Item | Budget |
|---|---|
| OS + toolchains (already there) | ~10 GB |
| Python env (`training/.venv`, torch CPU wheel) | ≤ 2 GB |
| Datasets under `$RL_DATA_DIR` (default `~/RinglessData`) | ≤ 6 GB |
| `training/runs/` (checkpoints, logs, previews) | ≤ 3 GB |

**Network:** the environment's default *Trusted* allowlist covers GitHub and
PyPI only. PyTorch CPU wheels and the datasets need the extra domains in §2.
GitHub (clone, push, release downloads) always works through the GitHub proxy.

**Domain rules that bind the model (from `research.md`):**
- **D17** total algorithmic latency = window + one hop ≤ **4 ms** at 48 kHz.
  Off-the-shelf 20–40 ms speech-enhancement models are forbidden as-is.
- **D18** train in the loop: phase 1 teacher-forced, phase 2 recursive
  in-loop fine-tune with a saturation stop (100 consecutive clipped samples),
  initialised from phase 1.
- **D19** report per-frame *streaming* cost, never offline real-time factor.
- **D20** inference engine is RTNeural: only layers RTNeural supports
  (Dense, Conv1D, GRU, LSTM, activations); no ONNX / LibTorch / Core ML in
  the audio callback; the exported JSON is the contract.
- **D21** speech / single voice only; Music mode stays notch-only.
- **Clean-room:** nothing from Alpha Labs (no binary, model, code, or
  decompilation), nothing from unlicensed research repos (Tencent AHS
  code) — papers only. Dependencies must be permissively licensed
  (MIT / BSD / Apache-2.0); record licences in `training/DATASETS.md`.

## 2. Owner prerequisites (one-time, Raf)

1. **Push the repo to a private GitHub repository** (this is the explicit
   "ok to push" that `00-overview.md` requires; the cloud clones from GitHub):
   ```sh
   cd ~/Workbench/Ringless
   git add ringlessmds/CLOUD-TRAINING.md ringlessmds/00-overview.md ringlessmds/03-advanced.md
   git commit -m "Stage3-cloud: add cloud training runbook"
   gh repo create innovatav/ringless --private --source . --remote origin --push
   ```
   Commit or stash the orchestrator's uncommitted C0 fixes first if they are
   still pending. Install the Claude GitHub App on the new repo (or run
   `/web-setup` once from a terminal `claude`).
2. **Create a cloud environment** named `ringless-train` at claude.ai/code →
   Environments:
   - Network access: **Custom**, include the default list, and add:
     ```text
     download.pytorch.org
     www.openslr.org
     openslr.org
     us.openslr.org
     zenodo.org
     huggingface.co
     *.huggingface.co
     *.hf.co
     ```
   - Environment variables:
     ```text
     BASH_DEFAULT_TIMEOUT_MS=600000
     BASH_MAX_TIMEOUT_MS=600000
     OMP_NUM_THREADS=4
     RL_DATA_DIR=/root/RinglessData
     RL_RUN=s-v1
     ```
   - Setup script (optional, after B0 exists; speeds up every session and is
     cached only if it finishes in ~5 min): `bash training/scripts/cloud_setup.sh`
3. **Start sessions in the right model.** Create each cloud session from
   claude.ai/code on repo `innovatav/ringless`, environment `ringless-train`,
   pick the model in the session's model picker (or type `/model opus`,
   `/model sonnet`, `/model haiku` as the first message), then paste the role's
   kickoff prompt from §3 verbatim. One role per session.
4. **Budget decision:** default chunk budget is 40 chunk-hours for `s-v1`
   (§9). Change `budget_hours` in the STATE block to override.

## 3. Roles and kickoff prompts

Each role is a separate cloud session. A session must confirm it is running
as the intended model class; if the owner started it on another model it says
so and stops.

### 3.1 Planner — Opus / Fable, exactly one session

Reads: this file (all); `00-overview.md` (Purpose, Non-Goals, Foundation);
`03-advanced.md` (C26–C29); `research.md` (D17–D21, Gaps); and these corpus
notes in `ringlessmds/corpus/structure/`: `arxiv-2409.10358-ultralow-latency-se.md`
(Wu & Braun — window size vs model size, asymmetric/learnable windows),
`arxiv-2307.08858-ultralow-latency-se.md` (2 ms hearing-aid SE),
`zhang-2023-recursive-ahs.md`, `zhang-2023-deep-ahs.md` (in-loop training,
saturation), `valin-2018-rnnoise.md` (small-GRU structure),
`chowdhury-2021-rtneural.md` (engine constraints),
`vanwaterschoot-2010-howling-detection-criteria.md` (howl physics, MSG).
Fetches the RTNeural source at the latest release tag and reads its layer
headers and `python/model_utils.py` before choosing layers; records the tag.

Produces, and nothing else:
- `training/DESIGN.md` with exactly these sections:
  1. Simulator spec (rooms, RIRs, PA delay, loop-gain law relative to MSG,
     clipping, noise, target definition, saturation stop, seeds).
  2. Front-end / features (window, hop, bands, normalisation).
  3. Architecture S and M (layer list with shapes, parameter counts from a
     script, latency in ms, which ops run in RTNeural and which in C++ post-
     processing).
  4. Loss (each term, weight, formula).
  5. Training schedule (phase 1 / phase 2 step counts, optimizer, LR schedule,
     batch, gradient clipping, and — after B3's probe — expected steps per
     25-minute chunk).
  6. Eval metrics and the G2c thresholds restated.
  7. Export contract → also written as `docs/model-format.md` (RTNeural JSON
     layer list and `model.json` sidecar schema that C30 will load).
  8. Dataset subset list with sizes and licences (from §5), and the disk sum.
  9. Deviations / open questions (Builders append here).
  10. Acceptance tests: what each pytest in §7 must assert, as a checklist.
- `training/configs/s.yaml`, `training/configs/m.yaml`, `training/configs/tiny.yaml`
  (the smoke-test config: ~5k params, 50 steps).

Rules: stay inside the bounds of §6 and give a one-line reason per choice;
write no pipeline code (a ≤ 30-line parameter-count script is fine); commit
`Stage3-cloud: design locked`, push `main`, set `design: locked <sha>` in STATE.

**Kickoff prompt (paste verbatim):**
```text
You are the PLANNER for Ringless's neural stage. Confirm you are running as an Opus-class or Fable model; if not, stop and tell the owner which model you are.
Read ringlessmds/CLOUD-TRAINING.md completely, then exactly the files its section 3.1 lists, in that order. Fetch the RTNeural source at its latest release tag and verify layer support before choosing any layer.
Deliver training/DESIGN.md (the ten sections of 3.1, nothing missing), training/configs/{s,m,tiny}.yaml and docs/model-format.md. Stay inside the bounds of section 6; give a one-line reason for every choice; verify parameter counts with a short torch script; write no pipeline code.
Commit as "Stage3-cloud: design locked", push main, update the STATE block in ringlessmds/CLOUD-TRAINING.md, and finish with the open questions for the owner.
```

### 3.2 Builder — Sonnet, one build step per session

Reads: this file (all), `training/DESIGN.md`, and the STATE block to find the
next unchecked step in §7. Does exactly that step. Runs the tests. Commits
`Stage3-cloud B<n>: <name>`, pushes `main`, ticks the step in STATE, reports.

Rules: follow DESIGN.md; a necessary deviation is recorded in DESIGN.md §9
before the code changes; wrap every command in `timeout 540`; nothing runs
longer than 25 minutes (use §8 for training); never commit data, `runs/` or
checkpoints to `main`; never touch `src/`, `01-mvp.md`, `LOOP.md` (owned by
the local MVP track).

**Kickoff prompt (paste verbatim):**
```text
You are a BUILDER for Ringless's neural stage. Confirm you are running as a Sonnet-class model; if not, stop and tell the owner which model you are.
Read ringlessmds/CLOUD-TRAINING.md completely, then training/DESIGN.md. In the STATE block find the first unchecked build step; section 7 defines it. Do exactly that step and nothing else. Follow DESIGN.md; if you must deviate, add a "Deviation" entry to DESIGN.md section 9 first.
Wrap every command in `timeout 540`; never run anything longer than 25 minutes. Run `uv run pytest -q` before committing. Commit as "Stage3-cloud B<n>: <name>", push main, tick the step in STATE, and report: files changed, each command with its exit code, and each done-when criterion as PASS or FAIL with one line of evidence.
```

### 3.3 Babysitter — Haiku, repeatable sessions after B5

Reads: only §8, §9 and the STATE block. Runs the §9 loop literally for at
most 4 chunks per session. Never edits code or configs.

**Kickoff prompt (paste verbatim):**
```text
You are the BABYSITTER for Ringless's training run. Confirm you are running as a Haiku-class model; if not, stop and tell the owner which model you are.
Read only sections 8 and 9 and the STATE block of ringlessmds/CLOUD-TRAINING.md. Follow section 9 literally: status, escalation check, one chunk, push, report; at most 4 chunks this session. Never edit any .py, .yaml or threshold. If an escalation rule fires, stop immediately and report with the template.
```

To make babysitting hands-free, the owner can create a claude.ai/code
**Routine** that starts a cloud session on `innovatav/ringless` every 3 hours
with the Babysitter prompt (model Haiku). It stops itself when STATE says
`gate G2c: PASS` or `blocked:` is set.

## 4. Repository layout for the training track

```
training/
  DESIGN.md                 Planner — the locked design (sections 1–10)
  DATASETS.md               B1 — every source: URL, licence, SHA256, size, split
  pyproject.toml            B0 — uv project, torch from the CPU index
  configs/{s,m,tiny}.yaml   Planner
  scripts/
    cloud_setup.sh          B0 — uv sync + fetch datasets; idempotent, < 5 min
    chunk.sh                B0 — one resumable training chunk (§8)
    ckpt_push.sh            B0 — publish runs/<run>/ to branch ckpt/<run> (§8)
    ckpt_pull.sh            B0 — restore runs/<run>/ from ckpt/<run>
    status.sh               B0 — one-screen run status for the Babysitter
  data/
    fetch.py                B1 — download, verify, resample to 48 kHz FLAC
    corpus.py               B1 — index + speaker-disjoint train/held-out split
    rir.py                  B1 — pyroomacoustics RIR generator + cache
    loop_sim.py             B1 — closed-loop feedback simulator + MSG estimate
    dataset.py              B1 — torch Dataset producing (mic, target) pairs
    preview.py              B1 — writes WAV pairs for listening
  model/
    fb.py                   B2 — window, analysis / synthesis bank, overlap-add
    net.py                  B2 — S / M / tiny networks per DESIGN.md
    streaming.py            B2 — frame-by-frame inference with carried state
  export/
    rtneural_json.py        B2 — RTNeural JSON + model.json sidecar
    reference_infer.py      B2 — numpy re-implementation of RTNeural math
  train.py                  B3 — chunked, resumable training loop
  eval/
    asg.py                  B4 — added-stable-gain via the Python loop sim
    metrics.py              B4 — SI-SDR, STOI, howl energy
    run.py                  B4 — evaluates a checkpoint → REPORT.md
    REPORT_TEMPLATE.md      B4
  tests/                    one file per step: test_env, test_sim, test_model,
                            test_export, test_train_smoke, test_eval
  runs/                     gitignored; lives on ckpt/<run> branches
docs/model-format.md        Planner (C27's contract for C30)
```

Mapping to the stage plan: C25 bench → the 2-minute steps/s probe in B3;
C26 → B1; C27 → B2 (+ `docs/model-format.md`); C28 → B3; C29 (Python half)
→ B4. Ownership rules of `00-overview.md` apply: the cloud track owns only
`training/`, `docs/model-format.md` and this file's STATE block.

## 5. Datasets — what, where, licence, size

All sources are openly licensed and fetched by `training/data/fetch.py`
into `$RL_DATA_DIR`; archives are deleted after extraction; everything is
resampled to 48 kHz mono FLAC. `DATASETS.md` records URL, licence, SHA256 and
the on-disk size of every item.

| Role | Source | Licence | Download | On disk | Notes |
|---|---|---|---|---|---|
| Clean speech — train | LibriSpeech `dev-clean` | CC BY 4.0 | `https://www.openslr.org/resources/12/dev-clean.tar.gz` (337 MB) | ~1.0 GB FLAC @48k | 16 kHz source upsampled: no content above 8 kHz. Accepted for `s-v1`; noted as a limitation in REPORT.md. |
| Clean speech — held-out | LibriSpeech `test-clean` | CC BY 4.0 | `https://www.openslr.org/resources/12/test-clean.tar.gz` (346 MB) | ~1.0 GB | Speaker-disjoint from dev-clean. Used only for validation and G2c. |
| Clean speech — optional | LibriSpeech `train-clean-100` | CC BY 4.0 | `.../12/train-clean-100.tar.gz` (6.3 GB) | ~18 GB | **Does not fit** the cloud budget. Local M1 Pro run only. |
| Clean speech — optional 48 kHz | VCTK 0.92 | CC BY 4.0 | one 10.9 GB zip | ~12 GB | Does not fit; local run only. (EARS and DAPS are 48 kHz but CC BY-**NC** — avoid.) |
| Noise | DEMAND (48 kHz) | CC BY-SA 4.0 | Zenodo record 1227121: query `https://zenodo.org/api/records/1227121` for the exact `<ENV>_48k.zip` file names; take `OMEETING`, `OHALLWAY`, `PCAFETER`, `PRESTO`, `SPSQUARE` | ~0.2 GB after keeping channel 1 only | Real venue-like rooms. If the record id has moved, search Zenodo for "DEMAND Diverse Environments Multichannel Acoustic Noise Database". |
| RIRs — primary | Synthetic, `pyroomacoustics` (MIT) | MIT | pip | ~0.2 GB cache | Shoebox rooms 40–2000 m³, RT60 0.2–1.2 s, 48 kHz, two RIRs per scene: talker→mic (0.05–0.6 m) and PA loudspeaker→mic (1–12 m). Unlimited, seeded, held-out seeds for eval. |
| RIRs — optional measured | openSLR 28 `RIRS_NOISES` | Apache-2.0 | `https://www.openslr.org/resources/28/rirs_noises.zip` (~1.3 GB) | ~0.5 GB real RIRs only | 16 kHz; off by default. |
| Howling | none — generated by the closed loop in `loop_sim.py` | — | — | — | Feedback is simulated, not recorded. |

Cloud total with the defaults: ≈ 2.5 GB. If `pip`/`uv` cannot reach a host,
the environment allowlist (§2) is missing it — report, do not work around.

## 6. Design defaults and bounds (Planner decides inside these)

| Item | Default for `s-v1` | Bound |
|---|---|---|
| Sample rate | 48 kHz | fixed |
| Window / hop | 96 / 48 samples (2 ms / 1 ms) → algorithmic latency 3 ms | window ≤ 192, hop ≤ 96, latency ≤ 4 ms (D17) |
| Front-end | learnable real filterbank: sqrt-Hann window → `Dense(96→N_BANDS, no bias)`; synthesis `Dense(N_BANDS→96)` + overlap-add; init from a DFT-like basis | N_BANDS ∈ {64, 96}; alternative: fixed 96-point STFT magnitude with a real mask |
| Body (S) | `GRU(N_BANDS→128)` → `GRU(128→128)` → `Dense(128→N_BANDS)` + sigmoid = mask | 150k–250k params; RTNeural layers only |
| Body (M) | same with 2×256 GRU | 0.6M–1.5M params |
| What runs where | RTNeural: analysis Dense → GRUs → mask Dense. C++ post: mask × analysis coefficients → synthesis Dense → OLA (matrices shipped in `model.json`) | no branch/multiply layers inside the RTNeural graph |
| Input extras | optional log-RMS of the frame appended (N_BANDS+1 inputs) | — |
| Level robustness | random input gain ±20 dB during training (no AGC in the app) | — |
| Target | direct + early (first 50 ms) talker speech at the mic, level-matched to the mic input | — |
| Loss | multi-resolution STFT (256/512/1024) spectral-convergence + log-magnitude L1, + 0.5·(−SI-SDR), + howl penalty (energy of narrowband peaks present in output but not in target) | weights tunable |
| Phase 1 | teacher-forced on simulated pairs | — |
| Phase 2 | recursive in-loop: unroll the model inside `loop_sim` on 2 s segments with truncated BPTT (100 ms), saturation stop at 100 consecutive clipped samples, init from the best phase-1 checkpoint | D18 |
| Optimizer | AdamW, lr 1e-3 cosine → 1e-5, batch 16 × 2 s, grad-clip 5, seed 1234 | — |
| Loop gain law | `g_rel` ∈ [−12, +3] dB relative to the scene's MSG (MSG from the PA RIR's peak magnitude for the identity processor); ≥ 30 % of samples above MSG so howling actually occurs | — |
| Validation set | 200 fixed pairs: test-clean speakers × held-out RIR seeds × 5 noise files | — |
| Runtime engine | RTNeural at its latest release tag, recorded in DESIGN.md §7 and `README.md` | D20 |

## 7. Build steps (Sonnet) — each with done-when

**B0 Environment and scaffold.**
`training/pyproject.toml` (uv project; Python 3.12; `torch` from
`https://download.pytorch.org/whl/cpu` declared as an explicit uv index so
the CPU wheel is used, never the CUDA bundle; `numpy`, `scipy`, `soundfile`,
`pyroomacoustics`, `pystoi`, `pyyaml`, `pytest`, `tqdm`); `.gitignore` entries
for `training/.venv/`, `training/runs/`, `training/data_cache/`; the five
scripts in §4 (`cloud_setup.sh`, `chunk.sh`, `ckpt_push.sh`, `ckpt_pull.sh`,
`status.sh`) implementing §8; `tests/test_env.py`.
*Done when:* `uv run python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
prints a CPU build and `False`; `du -sh training/.venv` ≤ 2 GB; `uv run pytest -q`
is green; running `cloud_setup.sh` twice in a row finishes the second time in
< 60 s; `ckpt_push.sh`/`ckpt_pull.sh` round-trip a dummy `runs/demo/` directory
through branch `ckpt/demo` (then delete the branch).

**B1 Data.** `fetch.py` (§5 sources, SHA256, resample, `DATASETS.md`),
`corpus.py`, `rir.py`, `loop_sim.py` (block-wise closed loop:
`mic = src + h_pa * spk`, `spk = clip(G · proc(mic)[n − d])`, uniformly
partitioned FFT convolution with partition = 96 samples, PA delay
d ∈ [2, 30] ms ≥ one partition, `msg_db(h_pa)` estimate, saturation guard),
`dataset.py` (on-the-fly, seeded, speaker-disjoint), `preview.py`.
*Done when:* `tests/test_sim.py` green — same seed ⇒ identical pair; `g_rel`
= +3 dB ⇒ last 0.5 s energy ≥ first 0.5 s + 20 dB and the guard trips;
`g_rel` = −6 dB ⇒ bounded; no held-out speaker in training; shapes
(2 s = 96000 samples, 48 kHz); `uv run python -m training.data.preview --n 5`
writes 5 WAV pairs that audibly ring (input) vs clean (target); throughput
≥ 20 pairs/s with 3 workers on the VM (else add a 1,000-pair precomputed cache).

**B2 Model and export.** `fb.py`, `net.py` (tiny/S/M from configs),
`streaming.py`, `export/rtneural_json.py`, `export/reference_infer.py`.
*Done when:* `tests/test_model.py` and `tests/test_export.py` green —
streaming output equals full-sequence output within 1e-4; parameter counts
within ±10 % of DESIGN.md §3; reported latency = window + hop ≤ 4 ms;
exported JSON re-run through `reference_infer.py` (numpy, RTNeural gate
ordering taken from the pinned source) matches torch within 1e-4; `model.json`
validates against `docs/model-format.md`.

**B3 Training loop.** `train.py --config --run --resume --max-minutes`:
step-based; checkpoint every 5 min and at exit (atomic write; keep `latest.pt`,
`best.pt`, previous `latest`); `metrics.jsonl` one line per 50 steps and per
validation (step, each loss term, val SI-SDR, lr, steps/s, elapsed, phase);
exits 0 when the minute budget is reached; handles SIGTERM; phase switch by
step count; `torch.set_num_threads(4)`; a `--probe` mode that runs 2 minutes
and prints steps/s. `tests/test_train_smoke.py` with `configs/tiny.yaml`.
*Done when:* smoke test green in < 3 min — 50 steps run, process killed,
`--resume` continues at step 51 with the optimizer state restored; the probe's
steps/s for `s.yaml` is written into DESIGN.md §5 with the resulting steps per
25-minute chunk.

**B4 Eval.** `eval/asg.py` (MSG binary search over G in the Python loop sim
for `proc = identity` and `proc = model`, 10 s runs, ≥ 20 held-out scenes,
ASG = MSG(model) − MSG(identity), report median and 10th percentile),
`eval/metrics.py` (SI-SDR; STOI via `pystoi` after downsampling to 16 kHz;
howl-energy ratio), `eval/run.py --ckpt … → runs/<run>/REPORT.md` from the
template with a PASS/FAIL line against §10. Eval must finish in ≤ 8 minutes on
the VM (cap scenes if needed and say so in the report).
*Done when:* `tests/test_eval.py` green — identity ASG = 0 ± 0.3 dB; a fixed
−6 dB gain "model" gives ASG = +6 ± 0.5 dB; `eval/run.py --ckpt random`
writes a REPORT.md that says FAIL.

**B5 First chunks.** Run three §8 chunks of `s.yaml` phase 1 (≈ 75 min of
wall clock across turns), push `ckpt/s-v1`, run `eval/run.py` once, write
`training/runs/README.md` (how to read `metrics.jsonl`, where checkpoints
live, how to resume locally with `--max-minutes 0`).
*Done when:* `ckpt/s-v1` has `latest.pt`, `best.pt`, `state.json`,
`metrics.jsonl`; validation loss decreased across the three chunks; STATE
shows the chunk count and metrics; the Babysitter can take over.

## 8. Training chunk protocol (resumable 25-minute chunks)

The VM cannot be trusted for longer than one chunk, so a run is a sequence
of identical chunks, each restoring from and publishing to the `ckpt/<run>`
branch. Checkpoints are small (S ≈ 3 MB with optimizer state, M ≈ 16 MB).

`scripts/chunk.sh <run> [minutes=25]` does, in order:
1. `ckpt_pull.sh <run>` — `git fetch origin ckpt/<run>` and restore
   `training/runs/<run>/` from it (no-op if the branch does not exist yet).
2. Launch in the background with a hard cap:
   `nohup timeout $((minutes*60+120)) uv run python train.py --config configs/<run>.yaml --run <run> --resume --max-minutes <minutes> > runs/<run>/chunk-$(date +%s).log 2>&1 &`
   The agent then polls with `sleep 240; tail -n 2 runs/<run>/metrics.jsonl`
   until the process has exited (never longer than 27 minutes in total).
3. `ckpt_push.sh <run>` — in a temporary git worktree, create or reset the
   **orphan** branch `ckpt/<run>` to a single commit containing `latest.pt`,
   `best.pt`, `state.json`, `metrics.jsonl` and any `REPORT.md`, then
   `git push --force origin ckpt/<run>`. Single-commit history keeps the
   repository small. Refuse to push any file > 50 MB.
4. Print the last `metrics.jsonl` line and the exit code.

Rules: one chunk per agent turn; never start a chunk with < 3 GB free (delete
old `chunk-*.log` files first); never run two chunks at once; `main` never
receives checkpoints. Running the same `train.py` locally with
`--max-minutes 0` is the unlimited (M1 Pro) mode.

## 9. Babysitting protocol (Haiku)

Per session, repeat at most 4 times:

1. `git pull --ff-only && bash training/scripts/status.sh $RL_RUN` → shows
   last step, last validation metrics, best metric, chunks completed, disk
   free, last chunk exit code, chunk-hours used vs `budget_hours`.
2. **Escalation check.** Stop the session and report (template below) if any
   holds; also set `blocked: <rule>` in STATE and push that one-line change:
   - last loss is NaN/inf, or exit code ≠ 0 for two consecutive chunks;
   - validation SI-SDR has not improved by ≥ 0.1 dB over the last 6 chunks
     (plateau → needs a Builder/Planner decision, not more chunks);
   - disk free < 3 GB after cleaning logs;
   - `ckpt_push.sh` failed, or any `uv run pytest -q` failure on a fresh clone;
   - chunk-hours used ≥ `budget_hours`;
   - STATE already says `gate G2c: PASS` or has a `blocked:` entry.
3. Otherwise `bash training/scripts/chunk.sh $RL_RUN 25`, poll as in §8,
   confirm the push, then append the result to STATE (`run: chunks N · step S ·
   val SI-SDR X · best Y`) and commit + push that STATE line only.
4. After every 8th chunk run `uv run python -m training.eval.run --ckpt best.pt`
   (≤ 8 min), push the REPORT with `ckpt_push.sh`, and copy the PASS/FAIL line
   into STATE. On PASS: set `gate G2c: PASS <sha>` and stop.

Never: edit `.py`, `.yaml`, thresholds or DESIGN.md; delete a `ckpt/*` branch;
push anything to `main` except the STATE line; relax a rule to keep going.

**Report template (end of every session):**
```text
RUN <run> — session report
chunks this session: N   total chunks: M   total chunk-hours: H / budget B
step: S   phase: 1|2   last val SI-SDR: X dB   best: Y dB (step Sb)
last ASG (if evaluated): median A dB, p10 A10 dB — gate G2c: PASS|FAIL|not run
disk free: D GB   last exit code: E   ckpt/<run> pushed: yes|no (<sha>)
escalation: none | <rule that fired>
next: continue | STOP — <reason>
```

## 10. Gate G2c and hand-back to the local track

The full C29 gate compares notch+NN against notch-only inside the C++
`FeedbackLoopSim`; the C++ notch engine (C7) is not available in the cloud, so
the cloud gate **G2c** is defined against the identity processor, on ≥ 20
held-out scenes (test-clean speakers, unseen RIR seeds, unseen DEMAND files):

| Metric | PASS threshold |
|---|---|
| Added stable gain, median: MSG(NN) − MSG(identity) | ≥ 3 dB |
| Added stable gain, 10th percentile | ≥ 1 dB |
| SI-SDR improvement at `g_rel` = −3 dB | ≥ 3 dB |
| STOI change at `g_rel` = −6 dB | ≥ −0.03 |
| Streaming equals full-sequence (test) | green |
| Algorithmic latency | ≤ 4 ms |
| Parameters (S) | ≤ 250k |

Thresholds are fixed by this file; no agent relaxes them. FAIL after the
budget is a legitimate outcome: the Babysitter stops, and the owner starts a
**Planner** session with the prompt "review runs/<run>/REPORT.md and
metrics.jsonl, diagnose, and write DESIGN v2" (data realism, capacity,
loss, schedule — in that order of suspicion).

**On PASS** (hand-back list):
1. Tag `main` as `model/s-v1`; the export (`model.rtneural.json`, `model.json`)
   is committed under `resources/models/s-v1/` (small files) on `main`.
2. Locally: C30 loads the export; C29's full gate (notch+NN vs notch-only,
   `ASG ≥ ASG(notch) + 4 dB`, CPU per channel on M1 Pro) runs once C7 exists;
   C31 wires the hybrid.
3. The M-size and long phase-2 runs use the same `train.py` on the M1 Pro
   with `--max-minutes 0` (MPS), per `03-advanced.md`.

## 11. Rules for every agent

- **Clean-room and licences** as in §1. New Python dependencies need a
  permissive licence and a line in `DATASETS.md` (section "Software").
- **Git:** code on `main`, weights on `ckpt/<run>`; commit messages start with
  `Stage3-cloud`; push after every step; never force-push `main`; never rewrite
  others' commits; never commit datasets, `runs/` or `.venv`.
- **Scope:** only `training/`, `docs/model-format.md`, `resources/models/`
  and this file's STATE block. `src/`, `tests/` (C++), `01-mvp.md`, `LOOP.md`
  belong to the local MVP track.
- **Time and disk:** every command under `timeout`; nothing > 25 minutes;
  checkpoint before any long operation; keep ≥ 3 GB free.
- **Honesty:** report numbers verbatim, FAIL included; never change a threshold
  to pass; say when a done-when criterion could not be checked in the cloud.
- **Blocked on the owner?** (allowlist, GitHub, budget, licence doubt): set
  `blocked:` in STATE, push, and stop. Do not work around a network block.

## STATE
- track: Stage 3 cloud (C25–C29 equivalents) — see `03-advanced.md` for C30–C32
- mode: single cloud session plays all roles on `rafikshehata/rafikshehata.github.io`, branch `claude/wizardly-hawking-ctkdaw` (= "main" here); checkpoints on `ckpt/<run>` (DESIGN §9 D0)
- design: locked ec6977b
- build: B0 ☑ (ckpt round-trip verified on a local bare remote only; GitHub push 403) · B1 ☑ · B2 ☑ · B3 ☑ · B4 ☑ · B5 ☐
- run s-v1: chunks 0 · step 0 · val SI-SDR — · best — · last ASG —
- budget_hours: 40
- gate G2c: pending
- blocked: GitHub push refused (Claude GitHub App lacks write access to the repo) — commits are local until the owner fixes access
- last update: 2026-10-18 — B0–B4 built, 50 tests green; B3 probe 0.655 / 0.088 steps/s (phase 1 / 2); eval 303 s; starting B5
