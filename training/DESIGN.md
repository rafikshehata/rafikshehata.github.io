# Ringless neural stage — DESIGN (s-v1)

Status: **locked** for `s-v1`. Builders follow this file; a necessary change is
recorded in §9 *before* the code changes. Binding inputs: `ringlessmds/CLOUD-TRAINING.md`
(§1 constraints, §5 datasets, §6 bounds, §10 gate) and the domain rules D17–D21
it restates. Every choice below carries a one-line reason (*why*).

Notation: `fs` = 48 000 Hz, `W` = window (samples), `H` = hop (samples),
`dB` = 20·log10 of an amplitude ratio unless stated otherwise.

---

## 1. Simulator spec

All randomness flows from integer seeds through `numpy.random.default_rng(seed)`;
the same seed always yields the same pair (bit-identical float32).

### 1.1 Rooms and RIRs (`data/rir.py`)

| Parameter | Value | Why |
|---|---|---|
| Room model | `pyroomacoustics.ShoeBox`, image-source method (ISM), order from `inverse_sabine(RT60)`, frequency-independent absorption, air absorption off | §5 primary RIR source; measured 0.01–0.8 s per RIR at 48 kHz on the VM, so full-order ISM is affordable for a cached bank. |
| Volume `V` | log-uniform 40–2000 m³ | §5 range; log spacing covers small meeting rooms and halls evenly. |
| Height `Hr` | uniform 2.7 – clamp(2.7 + V/400, 3.2, 8) m | Large rooms are taller; keeps aspect ratios plausible. |
| Floor | `A = V/Hr`, ratio `ρ ~ U[1, 2]`, `L = sqrt(A·ρ)`, `Wd = A/L` | Simple plausible rectangle. |
| RT60 | uniform 0.2–1.2 s; absorption clamped to ≤ 0.99 if Sabine is infeasible | §5 range. |
| Talker→mic | 0.05–0.6 m (uniform), random direction | §5; handheld / lectern mic. |
| PA→mic | 1–12 m (uniform), capped at 0.8 × room diagonal; positions ≥ 0.5 m from walls, heights 1.2–2.0 m (mic, talker) and 1.8–3.0 m (PA) | §5; PA cabinets are mounted high. |
| Directivity | omni for all transducers | pyroomacoustics default; limitation noted in REPORT. |
| PA band-limit | `h_pa` filtered by Butterworth HP (order 2, fc ~ U[80, 150] Hz) and LP (order 4, fc ~ U[12, 18] kHz), zero-phase not used (causal `sosfilt`) | Real PA cabinets roll off; prevents unrealistic howl at 20 kHz. |
| Truncation | at the EDC −60 dB point, capped at 1.0 s (`h_pa`) and 0.5 s (`h_talk`); 5 ms half-Hann fade-out | Keeps the cache small; the cut tail is ≥ 50 dB down. |
| Bank | 2 000 training scenes (seeds 0–1999), 200 held-out scenes (seeds 1 000 000–1 000 199), cached as float16 `.npz` shards under `$RL_DATA_DIR/rir/` | Precomputed once (≈ 3 min on 4 cores); held-out seeds never appear in training. |

### 1.2 Closed loop (`data/loop_sim.py`)

```
mic[n] = s_mic[n] + v[n] + (h_pa * spk)[n]
spk[n] = clip_c( G · p[n − d] )          clip_c(x) = min(max(x, −c), c)
p      = proc(mic)                        (processor output stream)
```

| Item | Value | Why |
|---|---|---|
| Convolution | uniformly partitioned overlap-save FFT convolution, partition `B` = 96, FFT 192 | §7 B1; 96 ≤ smallest PA delay so the loop is computable block by block. |
| PA delay `d` | integer samples, uniform 96–1440 (2–30 ms) | §7 B1 range; ≥ one partition. |
| Normalisation | `h_pa` is scaled so `max_f |H_pa(f)| = 1` (FFT length = next pow2 ≥ 4·len) → identity MSG estimate = 0 dB and `G = 10^(g_rel/20)` | MSG from the PA RIR's peak magnitude (§6); makes `g_rel` the gain directly. |
| `msg_db(h)` | `−20·log10(max_f |H(f)|)` on the un-normalised RIR, stored in the bank index | §7 B1 MSG estimate. |
| Gain law | `g_rel ~ U[0, +3]` with prob 0.35, else `U[−12, 0]` (dB) | §6 bound [−12, +3] with ≥ 30 % above MSG. |
| Clip level | `c = rms(s_mic) · 10^(hr/20)`, headroom `hr ~ U[18, 30]` dB | Speech at the PA sits 18–30 dB below clip at `g_rel` = 0, so stable scenes rarely clip and howls do. |
| Processors | `tf` teacher-forced: `p[n] = s_tgt[n − 48]` (open loop, one FFT convolution); `id` identity: `p = mic`; `model`: streaming network (stream delay 48 samples, §2); `gain(a)`: `p = a·mic` (tests) | `tf` mimics a perfect processor with the model's delay; `id` creates real howling. |
| Saturation guard | a sample is *clipped* when `|G·p[n−d]| ≥ c`; the guard **trips when ≥ 100 clipped samples fall inside any 480-sample (10 ms) window**; after a trip the PA is muted (`spk = 0`) for the rest of the segment ("protection mute") and the trip sample is recorded | See §9 D1: a clipped sinusoidal howl above ~240 Hz never produces 100 *consecutive* clipped samples. Muting keeps segments fixed-length without loss masks. |
| Segment | 2.0 s = 96 000 samples for training pairs; 10 s for eval | §7 B1 shapes; §7 B4 eval length. |

### 1.3 Pair recipe (`data/dataset.py`)

1. Draw `scene` (from the bank), speech crop (2 s from a random utterance of a
   split's speaker; utterances shorter than 2 s are zero-padded at the end), noise
   crop, `g_rel`, `d`, `hr`, input level, SNR, mode — all from the pair seed.
2. `s_mic = speech * h_talk` (full RIR); **target** `s_tgt = speech * h_early`
   where `h_early = h_talk[: n_dir + 2400]` (direct path index `n_dir` + 50 ms)
   with a 5 ms half-Hann fade-out. *Why:* §6 target = direct + first 50 ms.
3. Level: scale speech so `rms(s_mic)` = input level `~ U[−55, −15]` dBFS (−35 ± 20 dB);
   `s_tgt` gets the same factor (level-matched). *Why:* §6 ±20 dB level robustness, no AGC.
4. Noise `v`: DEMAND channel-1 crop scaled to SNR `~ U[5, 30]` dB w.r.t. `rms(s_mic)`.
   Noise is **not** in the target. *Why:* venue noise is part of the input only.
5. Mode: `tf` with prob 0.7, `id` with prob 0.3 (phase 1). `id` pairs use
   `g_rel ~ U[−2, +3]` so they contain ringing or howl build-up then the
   protection mute. *Why:* teacher forcing alone never howls; §6 requires
   howling to actually occur.
6. Output `(mic, s_tgt)` float32, both 96 000 samples; plus metadata (g_rel,
   tripped, trip sample, scene id) for logging.

Splits: speech train = LibriSpeech `dev-clean`, held-out = `test-clean`
(speaker-disjoint, asserted by `corpus.py`). Noise: each DEMAND file's first
240 s train, last 60 s held-out (see §9 D4). RIRs: seed ranges above.

**Synthetic source** (`data.source: synthetic`, used by `tiny.yaml` and tests):
harmonic pulse-train "speech" (f0 80–250 Hz, 3 formant resonators, 4 Hz
syllabic envelope, pauses), white-ish noise, and small ISM rooms generated on
the fly. *Why:* tests and the smoke run must pass on a fresh clone without datasets.

### 1.4 Phase-2 loop

The streaming model runs inside the closed loop block by block (2 hops per
96-sample partition), batch-parallel, `g_rel ~ U[−3, +3]` with 50 % above 0,
saturation guard + protection mute as above. See §5 for gradients.

---

## 2. Front-end / features

| Item | Value | Why |
|---|---|---|
| Window / hop | `W` = 96 (2 ms), `H` = 48 (1 ms) | §6 default; algorithmic latency **W + H = 144 samples = 3.0 ms ≤ 4 ms** (D17). |
| Analysis | `c = A·x`, `A ∈ R^{96×96}` learnable, no bias; `x` = the newest 96 input samples | Learnable real filterbank (§6), complete (96 outputs for 96 inputs). |
| `A` init | rows 0–48 = `w_a[n]·cos(2πkn/96)` (k = 0…48), rows 49–95 = `−w_a[n]·sin(2πkn/96)` (k = 1…47); `w_a` = periodic sqrt-Hann | DFT-like init (§6); makes the start point an exact STFT. |
| Bin pairing | bin 0 ↔ row 0; bin k ∈ 1…47 ↔ rows (k, 48 + k); bin 48 ↔ row 48 → **49 bins** | Pairs act like Re/Im so a phase-invariant power exists per bin. |
| Features | `P_k = Σ_{rows of bin k} c²`; `f_k = 0.5·(log10(P_k + 1e−10) + 5)` → 49 inputs | Log power is level-shift-friendly and phase-invariant; RTNeural has no square/log, so this runs in C++ (§3). |
| Log-RMS extra | not used | Redundant: log power already carries level. |
| Mask | 49 sigmoid outputs, each applied to all rows of its bin → `m_full ∈ (0,1)^96` | Real mask per bin (§6 alternative form). |
| Synthesis | `z = S·(m_full ⊙ c)`, `S ∈ R^{96×96}` learnable, init = periodic sqrt-Hann × inverse real DFT scaled for perfect reconstruction at 50 % overlap | `A0`, `S0` reconstruct exactly with OLA (Hann² sums to 1). |
| OLA | accumulate `z` into a 96-sample buffer, emit its first 48 samples, shift by 48 | Standard hop-48 overlap-add. |
| Stream delay | emitted sample `out[m] = y_full[m − 48]` where `y_full` is the full-sequence estimate aligned with the target | Exact equality streaming ⇔ full-sequence (tested to 1e−4). |
| Normalisation | none beyond the fixed log affine above | GRU input weights absorb any per-feature affine map. |

---

## 3. Architecture S and M

Graph per frame (1 000 frames/s):

```
C++ pre   : x(96) ─A→ c(96) ─pair-power, log affine→ f(49)
RTNeural  : f(49) → GRU(49→Hd1) → GRU(Hd1→Hd2) → Dense(Hd2→49) + sigmoid → m(49)
C++ post  : m(49) ─expand→ m_full(96) ⊙ c(96) ─S→ z(96) ─OLA→ 48 samples out
```

No branch or multiply layer exists inside the RTNeural graph (D20, §6 bound);
the filterbank matrices and feature constants ship in `model.json` (§7).
*Why the analysis moved out of RTNeural:* the phase-invariant power feature
needs squaring, which no RTNeural layer provides (§9 D2).

| Model | Layers (input → output) | Params (script) | Latency |
|---|---|---|---|
| **S** | GRU 49→128 (68 736), GRU 128→128 (99 072), Dense 128→49 (6 321), A + S (18 432) | **192 561** | 3.0 ms |
| **M** | GRU 49→256 (235 776), GRU 256→256 (394 752), Dense 256→49 (12 593), A + S (18 432) | **661 553** | 3.0 ms |
| tiny | GRU 49→16 (3 216), GRU 16→16 (1 632), Dense 16→49 (833), filterbank frozen | **5 681** trainable | 3.0 ms |

Parameter counts from `uv run python scripts/param_count.py` (torch layer
counts; GRU = 3·(in·h + h·h + 2h), PyTorch/RTNeural two-bias form).
S sits inside 150k–250k (§6) and ≤ 250k (G2c); M inside 0.6M–1.5M.

Mask Dense bias initialised to +3.0 (sigmoid ≈ 0.95). *Why:* the untrained
model starts close to the identity processor, which keeps early phase-2 loops sane.

Per-frame streaming cost (D19), multiply-accumulates per 1 ms frame:

| Block | S | M |
|---|---|---|
| Analysis `A·x` | 9 216 | 9 216 |
| Features | ~150 | ~150 |
| GRU 1 | 67 968 | 234 240 |
| GRU 2 | 98 304 | 393 216 |
| Dense | 6 272 | 12 544 |
| Synthesis `S·(m⊙c)` | 9 216 | 9 216 |
| **Total** | **≈ 191k MAC/frame ≈ 0.19 GMAC/s** | **≈ 659k MAC/frame ≈ 0.66 GMAC/s** |

CPU time per frame is measured locally in C30 (needs the M1 Pro).

---

## 4. Loss

`L = L_mr + 0.05 · (−SI-SDR_dB) + 1.0 · L_howl`, computed on full 2 s
sequences of `ŷ = y_full` against `s = s_tgt`.

| Term | Formula | Weight | Why |
|---|---|---|---|
| `L_mr` | mean over (n_fft, hop) ∈ {(256, 64), (512, 128), (1024, 256)}, Hann: `SC = ‖|S|−|Ŝ|‖_F / ‖|S|‖_F` + `LM = mean |log(|S|+δ) − log(|Ŝ|+δ)|`, with `δ = 1e−3 · max|S|` per item | 1.0 | §6; δ relative to the target bounds the log range to 60 dB so speech pauses do not dominate. |
| SI-SDR | zero-mean scale-invariant SDR in dB over the sequence, eps 1e−8 | 0.05 per dB | §6's 0.5 weight applied per 10 dB, which keeps it the same order as `L_mr`. |
| `L_howl` | STFT 2048/512 Hann power `PY`, `PS`; local mean `AY` = 31-bin average over frequency; peak weight `w = σ((10·log10((PY+ε)/(AY+ε)) − 6)/1.5)`; excess `E = relu(PY − 2·PS)`; `L_howl = log(1 + Σ w·E / (Σ PS + ε))` | 1.0 | §6 "energy of narrowband peaks present in output but not in target", scale-invariant and log-bounded. |

---

## 5. Training schedule

| Item | Phase 1 | Phase 2 | Why |
|---|---|---|---|
| Data | §1.3 pairs, 70 % `tf` / 30 % `id`, on the fly, 3 workers | model in the loop (§1.4), generated by the current weights each step | D18. |
| Init | random (seed 1234), mask bias +3 | best phase-1 checkpoint by val SI-SDR | D18. |
| Optimizer | AdamW, β (0.9, 0.999), weight decay 0.01 on matrices only | same, fresh moments | §6. |
| LR | 1e−3 → 1e−5 cosine over phase-1 steps, 500-step linear warm-up | 3e−4 → 1e−5 cosine over phase-2 steps | §6; a lower restart for fine-tuning. |
| Batch | 16 × 2 s (2 ranks × 8, §9 B3-2) | 16 × 6 s rollouts (§9 B3-3) | §6. |
| Gradient clip | global norm 5 | 5 | §6. |
| BPTT | full 2 s sequence | GRU state detached every 100 frames (100 ms); no gradient through the acoustic feedback path (§9 D3) | §6 truncated BPTT; keeps cost bounded. |
| Saturation stop | n/a (`id` pairs mute on trip) | guard trip → protection mute; trip rate logged | D18. |
| Threads | `torch.set_num_threads(4)` main process; workers single-threaded | same | §7 B3. |
| Checkpoint | every 5 min and at exit; atomic; `latest.pt`, `prev.pt`, `best.pt` | same | §7 B3, §8. |
| Logging | `metrics.jsonl` every 50 steps + each validation | same | §7 B3. |
| Validation | every 500 steps and at exit: 200 fixed held-out pairs (§6) → val loss, val SI-SDR | every 100 steps, same set | Comparable across phases. |

**Step counts.** Budget = 40 chunk-hours = 96 chunks of 25 min. Measured by
B3's probe (`train.py --probe`, 4 ranks × 1 thread, 1 data worker each, VM idle):

| Probe (B3) | steps/s | steps per 25-min chunk | phase steps | chunks |
|---|---|---|---|---|
| S phase 1 (16 × 2 s) | 0.655 | 923 | **20 000** (was 40 000, §9 B5-2) | ≈ 22 |
| S phase 2 (16 × 6 s rollouts) | 0.088 | 124 | **8 000** (was 5 500, §9 B5-2) | ≈ 65 |

(2 ranks gave 0.532 steps/s in phase 1; 1 process with 4 threads and 3 workers
gave 0.143.) The split first moved from the planned 60/30 chunks to ≈ 43/44
because a phase-2 step costs 7.4× a phase-1 step and phase 2 is the in-loop
training the gate measures, then to ≈ 22/65 at step 10 101 (§9 B5-2); the
rest of the 96 chunks is spare for evals and restarts.
Validation runs every 500 steps in phase 1 (≈ 13 min) and every 100 in phase 2
(≈ 19 min), plus at every exit.

---

## 6. Eval metrics and gate G2c

Eval set: **24 held-out scenes** (RIR seeds 1 000 000–1 000 023, test-clean
speakers, held-out DEMAND segments), 10 s of concatenated speech per scene,
input level −35 dBFS, SNR 20 dB, headroom 24 dB, `d` from the scene seed.

* **Stability at gain G** (10 s run): stable ⇔ the guard never trips **and**
  the feedback-to-source ratio over the last 2 s,
  `10·log10(E[(mic − s_mic − v)²] / E[(s_mic + v)²])`, is ≤ +10 dB.
* **MSG(proc)**: binary search over `g_rel` ∈ [−12, +24] dB, 9 iterations
  (0.07 dB resolution), all scenes batched; assumes stability is monotone in G.
* **ASG** = MSG(model) − MSG(identity) per scene → median and 10th percentile.
* **SI-SDRi** at `g_rel` = −3 dB: SI-SDR(model loop output, aligned by 48
  samples, vs `s_tgt`) − SI-SDR(identity loop mic vs `s_tgt`), first 0.5 s
  excluded; median over scenes.
* **ΔSTOI** at `g_rel` = −6 dB: STOI(model output) − STOI(identity mic), both
  vs `s_tgt`, after resampling to 16 kHz (`pystoi`, non-extended); median.
* **Howl-energy ratio** (reported, not gated): `L_howl`'s energy ratio for the
  model loop vs the identity loop at `g_rel` = 0 dB.
* **Level offset** (reported, not gated): `10·log10(E[y²]/E[s_tgt²])` at
  `g_rel` = −12 dB, so a model that "wins" ASG by attenuating everything is visible.

G2c thresholds (fixed by CLOUD-TRAINING.md §10, restated, never relaxed):

| Metric | PASS |
|---|---|
| ASG median | ≥ 3 dB |
| ASG 10th percentile | ≥ 1 dB |
| SI-SDRi at −3 dB | ≥ 3 dB |
| ΔSTOI at −6 dB | ≥ −0.03 |
| Streaming = full-sequence test | green |
| Algorithmic latency | ≤ 4 ms (design: 3.0 ms) |
| Parameters (S) | ≤ 250k (design: 192 561) |

---

## 7. Export contract

Full text: `docs/model-format.md` (the contract C30 loads). Summary:

* `model.rtneural.json` — RTNeural `json_parser::parseJson` format,
  `in_shape [null, null, 49]`, layers `gru` (Hd1), `gru` (Hd2),
  `dense` 49 with `"activation": "sigmoid"`. GRU weights are
  `[kernel (in × 3h), recurrent (h × 3h), bias (2 × 3h)]` with gate column
  order **z | r | h** and bias row 0 = input bias, row 1 = recurrent bias
  (verified in RTNeural `RTNeural/gru/gru.h` and `model_loader.h`). Dense
  weights are `[kernel (in × out), bias (out)]`.
* `model.json` — sidecar with sample rate, W, H, latency, the 96×96 analysis
  and synthesis matrices (windows folded in), bin pairing, feature constants,
  stream delay, RTNeural file name + SHA-256, the pinned RTNeural commit,
  parameter counts and training provenance.
* **Pinned RTNeural:** repository `jatinchowdhury18/RTNeural` has **no
  release tags** (checked 2026-10-18 via `git ls-remote --tags`: none;
  `CMakeLists.txt` says `VERSION 1.0.0`). Pinned instead to `main` commit
  `95c3c0f987a6fe903e7eec71e797405dbed7caf7` (2026-08-20, "Update XSIMD
  submodule to version 14.3.0"). See §9 D5.

---

## 8. Datasets (subset, sizes, licences, disk)

| Item | Source | Licence | Download | On disk (est.) |
|---|---|---|---|---|
| Speech train | LibriSpeech `dev-clean` (openslr 12) | CC BY 4.0 | 337 MB | ~1.0 GB FLAC @ 48 kHz |
| Speech held-out | LibriSpeech `test-clean` | CC BY 4.0 | 346 MB | ~1.0 GB |
| Noise | DEMAND 48 kHz (Zenodo 1227121): OMEETING 253 MB, OHALLWAY 253 MB, PCAFETER 314 MB, PRESTO 327 MB, SPSQUARE 321 MB — channel 1 kept | CC BY 4.0 (Zenodo record metadata; see §9 D6) | 1 468 MB (deleted after extraction) | ~0.1 GB FLAC |
| RIR bank | pyroomacoustics (MIT), 2 200 scenes, float16 | MIT (tool); generated data | — | ~0.3 GB |
| **Total kept** | | | ~2.2 GB downloads | **≈ 2.4 GB ≤ 6 GB** |

Peak during fetch: + 1.5 GB of archives briefly (one DEMAND zip at a time, so
≤ 0.4 GB extra in practice). `training/DATASETS.md` (B1) records URL, licence,
SHA-256 and measured size of every item, plus a "Software" section.

---

## 9. Deviations / open questions

Planner decisions that depart from the letter of CLOUD-TRAINING.md:

* **D0 — single session, this repository.** The owner chose to run every role
  in one cloud session on `rafikshehata/rafikshehata.github.io`, branch
  `claude/wizardly-hawking-ctkdaw`, instead of separate Opus/Sonnet/Haiku
  sessions on `innovatav/ringless` `main`. "`main`" in the runbook reads as this
  branch; checkpoints still go to orphan branch `ckpt/<run>`. The corpus notes,
  `00-overview.md`, `03-advanced.md` and `research.md` are **not in this
  repository**; this design relies on the rules the runbook restates (§1, §6,
  §10) and on the papers by title only.
* **D1 — saturation guard.** "100 consecutive clipped samples" cannot fire for a
  clipped sinusoidal howl above ~240 Hz (the clipper releases every half
  period). Implemented as ≥ 100 clipped samples within any 10 ms window, plus a
  protection mute after the trip.
* **D2 — what runs where.** The analysis matrix runs in C++ pre-processing,
  not as an RTNeural Dense, because the phase-invariant power feature needs
  squaring. RTNeural still runs GRU → GRU → Dense + sigmoid; no branch/multiply
  inside the graph.
* **D3 — phase-2 gradients.** Inputs are generated in the loop by the current
  model (no gradient), then the model is trained on those inputs with 100 ms
  truncated BPTT on its own recurrence. Back-propagating through the acoustic
  feedback path would need ~1 GB of autograd state per step on this VM.
* **D4 — "unseen DEMAND files".** Only five DEMAND files are fetched, so
  held-out noise is the last 60 s of each file (time-disjoint), not separate files.
* **D5 — RTNeural has no release tags.** Pinned to a `main` commit (§7).
* **D6 — DEMAND licence.** The runbook lists CC BY-SA 4.0; the Zenodo record's
  metadata says CC BY 4.0. Recorded as CC BY 4.0 with the record id.
* **D7 — Python 3.12 via uv.** The VM ships 3.13; uv installs 3.12.13 as §7 B0 asks.
* **D8 — network.** PyTorch CPU wheels are served from `download-r2.pytorch.org`,
  which the §2 allowlist does not include; the owner must add it.

Open questions for the owner:

1. Phase 2 trains at `g_rel` ≤ +3 dB (the §6 bound) while G2c needs a median
   ASG ≥ 3 dB, so training stops exactly at the gate edge. Allow phase 2 up to
   +6 dB for margin?
2. 40 training speakers (dev-clean) is little speaker variety; train-clean-100
   does not fit the cloud disk. Accept for s-v1?
3. The gate counts global attenuation as stable gain (a −6 dB gain gives
   ASG = +6). The report adds a level-offset line; should a future gate correct ASG for it?

Builder deviations are appended below this line.

* **B1-1 — identity pairs are simulated for 6 s.** Measured on 24 held-out rooms
  (identity loop, headroom 24 dB): guard trip rate within 2 s was 0.04 / 0.12 /
  0.33 at `g_rel` = +1 / +2 / +3 dB, but 0.21 / 0.71 / 0.92 within 6 s (median
  trip at 4.3 / 3.4 / 2.0 s). Two-second cold starts therefore rarely howl. Each
  `id` pair now runs 6 s (`sim.id_sim_seconds`) and keeps the 2 s window ending
  100 ms after the trip (the last 2 s if it never trips). `g_rel` stays in §6 bounds.
* **B1-2 — identity-pair cache.** With B1-1 the on-the-fly throughput fell to
  11.8 pairs/s with 3 workers (< 20). As §7 B1 prescribes, 4 000 `id` pairs are
  precomputed once into `$RL_DATA_DIR/idcache/` (float16 with per-pair peak
  scaling, 1.5 GB) and drawn by seed with a random ±6 dB gain and polarity;
  teacher-forced pairs and all validation pairs stay on the fly.
* **B1-3 — teacher-forced pairs skip the guard.** They are open loop (the PA
  plays the clean target), so there is no runaway to stop; clipping still applies.
* **B1-4 — preview listening.** "Audibly ring" cannot be judged by ear in the
  cloud; `preview.py` prints trip status and level growth for each pair instead.
* **B1-5 — long simulations concatenate speech.** Segments longer than 2.5 s
  (6 s identity pairs, phase-2 rollouts, 10 s eval scenes) use back-to-back
  utterances of one speaker. A single crop ran past short utterances into zero
  padding and produced near-silent targets (found in B5 chunk 1: loss 2e5; the
  cache and validation set were rebuilt and the run restarted). The
  spectral-convergence term is also clamped at 10 per item.
* **B2-1 — parameter counts are exact.** `RinglessNet` counts equal §3 (S 192 561,
  M 661 553, tiny 5 681), so the ±10 % tolerance is not used.

* **B3-1 — no weight decay on the filterbank.** AdamW decay applies to GRU and
  Dense weight matrices only; decaying `A`/`S` would erode the DFT initialisation.
* **B3-2 — two single-thread data-parallel ranks.** Measured on the VM for S,
  batch 16 × 2 s: 2.75 s/step with 1 thread, 2.29 s/step with 4 threads (GRU
  recursion is sequential), and 7.0 s/step with 4 threads plus 3 data workers
  (oversubscription). `train.procs: 2` runs two ranks (gloo all-reduce, 8 items
  each, 1 thread each, 2 data workers each) for the same global batch of 16;
  `torch.set_num_threads(train.threads)` still applies when `procs` is 1.
  Resume stays bit-exact for a fixed rank count.
* **B3-3 — phase-2 rollouts are 6 s.** For the reason in B1-1, the model runs
  in the loop for 6 s (`phase2.rollout_seconds`) and is trained on the whole
  rollout with 100-frame truncated BPTT.
* **B3-4 — tiny config.** 40 + 20 = 60 steps so the smoke test can stop at step
  50 and resume at 51; one thread so it stays fast while other jobs run.
* **B4-1 — eval details.** Eval scenes concatenate one held-out speaker's
  utterances to 10 s (test-clean utterances are mostly shorter); eval runs
  single-threaded by default (same oversubscription finding as B3-2).

* **B5-2 — phase switch moved earlier (Planner, step 10 101).** Evidence: G2c
  evals at steps 3 022 / 6 655 / 10 000 gave ASG median +5.41 / +6.08 / +6.26 dB
  (p10 ≥ +4.06, passing) but SI-SDRi at −3 dB of only +1.82 / +2.08 / +2.16 dB,
  gaining ≈ 0.1 dB per four chunks. A diagnosis on 6 eval scenes showed the model
  reaches 13.9 dB SI-SDR open loop on talker + noise but 11.4 dB inside the loop at
  −3 dB: the loss is in-loop, which phase 2 trains on. `phase1_steps` 40 000 →
  20 000 and `phase2_steps` 5 500 → 8 000 (≈ 8.5 + 25.3 = 33.8 chunk-hours of
  the 40). Thresholds unchanged. The phase-1 cosine now ends at step 20 000, so
  the learning rate steps down from 8.5e-4 to 5.0e-4 at chunk 14.

---

## 10. Acceptance tests (checklist)

`tests/test_env.py` (B0)
- [ ] torch imports, `torch.cuda.is_available()` is False, version string has `+cpu` or no CUDA build.
- [ ] numpy, scipy, soundfile, pyroomacoustics, pystoi, yaml import.
- [ ] the three configs load and `s.yaml` parameter budget fields match §3.
- [ ] scripts exist and are executable; `bash -n` passes on each.

`tests/test_sim.py` (B1)
- [ ] same seed ⇒ identical `(mic, target)` (`np.array_equal`); different seed ⇒ different.
- [ ] shapes: 96 000 samples, float32, fs 48 000.
- [ ] identity loop at `g_rel` = +3 dB: energy of the last 0.5 s before the trip ≥ first 0.5 s + 20 dB, and the guard trips.
- [ ] identity loop at `g_rel` = −6 dB: no trip, final 0.5 s RMS ≤ 10 × the source RMS (bounded).
- [ ] partitioned convolution equals `scipy.signal.fftconvolve` within 1e−5 (open loop).
- [ ] `msg_db` matches a brute-force FFT maximum within 0.01 dB.
- [ ] corpus split: no held-out speaker id in the training index (synthetic tree always; real data when present).
- [ ] guard: ≥ 100 clipped samples in 10 ms trips, 99 do not.

`tests/test_model.py` (B2)
- [ ] streaming output equals full-sequence output within 1e−4 (S and tiny, random weights, 0.3 s input).
- [ ] parameter counts within ±10 % of §3 (exact expected: S 192 561, M 661 553, tiny 5 681 trainable).
- [ ] reported latency = W + H = 144 samples = 3.0 ms ≤ 4 ms.
- [ ] with the initial filterbank and mask ≡ 1, the model reconstructs its input delayed by 0 (full) / 48 (stream) within 1e−4.

`tests/test_export.py` (B2)
- [ ] `reference_infer.py` (numpy, RTNeural gate order z|r|h, bias rows input/recurrent) on the exported JSON matches torch masks and audio within 1e−4.
- [ ] `model.json` validates against `docs/model-format.md` (required keys, shapes, sha256 of the RTNeural file).
- [ ] RTNeural JSON layer list is exactly gru, gru, dense(sigmoid) with the right shapes.

`tests/test_train_smoke.py` (B3)
- [ ] `tiny.yaml` 50 steps run, process is killed (SIGTERM) after a checkpoint, `--resume` continues at step 51 with optimizer state and RNG state restored; whole test < 3 min.
- [ ] resume is bit-exact: two runs (straight vs. killed+resumed) give identical weights after the same step count.
- [ ] `metrics.jsonl` lines have the required keys.

`tests/test_eval.py` (B4)
- [ ] identity ASG = 0 ± 0.3 dB.
- [ ] fixed −6 dB gain processor ASG = +6 ± 0.5 dB.
- [ ] `eval/run.py --ckpt random` writes a REPORT.md containing `FAIL`.
