# REPORT — s-v1

- checkpoint: `best_gate.pt` (step 17856, phase 2)
- config: `configs/s.yaml` · git `a2164798e545` · generated 2026-10-18 22:21 UTC
- eval: 24 held-out scenes × 10.0 s, search [-12, 24] dB in 9 steps, wall time 205.1 s

## Gate G2c: FAIL

| Metric | Value | Threshold | Result |
|---|---|---|---|
| ASG median | +5.94 dB | ≥ 3 dB | PASS |
| ASG 10th percentile | +4.19 dB | ≥ 1 dB | PASS |
| SI-SDR improvement at −3 dB (median) | +2.48 dB | ≥ 3 dB | FAIL |
| STOI change at −6 dB (median) | +0.010 | ≥ −0.03 | PASS |
| Streaming = full sequence | max err 6.3e-08 | ≤ 1e-4 (test green) | PASS |
| Algorithmic latency | 3.0 ms | ≤ 4 ms | PASS |
| Parameters | 192,561 | ≤ 250k | PASS |

## Per-scene added stable gain (dB re identity MSG estimate)

| Scene | MSG identity | MSG model | ASG | Flags |
|---|---|---|---|---|
| 0 | +0.59 | +7.27 | +6.68 |  |
| 1 | +2.20 | +9.38 | +7.17 |  |
| 2 | +2.98 | +10.50 | +7.52 |  |
| 3 | +0.66 | +9.30 | +8.65 |  |
| 4 | +2.06 | +4.38 | +2.32 |  |
| 5 | +1.57 | +5.86 | +4.29 |  |
| 6 | +2.13 | +8.11 | +5.98 |  |
| 7 | +2.20 | +7.27 | +5.06 |  |
| 8 | +2.27 | +7.90 | +5.62 |  |
| 9 | +1.71 | +8.74 | +7.03 |  |
| 10 | +1.08 | +6.77 | +5.70 |  |
| 11 | +2.20 | +8.32 | +6.12 |  |
| 12 | +1.71 | +7.62 | +5.91 |  |
| 13 | +1.99 | +7.27 | +5.27 |  |
| 14 | +1.92 | +8.46 | +6.54 |  |
| 15 | +0.94 | +7.90 | +6.96 |  |
| 16 | +1.43 | +5.93 | +4.50 |  |
| 17 | +3.61 | +9.59 | +5.98 |  |
| 18 | +0.80 | +5.09 | +4.29 |  |
| 19 | +0.30 | +5.51 | +5.20 |  |
| 20 | +2.20 | +5.65 | +3.45 |  |
| 21 | +2.13 | +8.74 | +6.61 |  |
| 22 | +1.64 | +5.79 | +4.15 |  |
| 23 | +1.85 | +8.11 | +6.26 |  |

## Reported, not gated

| Metric | Value |
|---|---|
| Median SI-SDR at −3 dB, identity / model | 9.43 / 12.20 dB |
| Howl-energy ratio at 0 dB, identity / model | 0.051 / 0.003 |
| Level offset at −12 dB (model output vs target) | -0.67 dB |
| Median MSG identity / model (re estimate) | +1.89 / +7.76 dB |

## Notes

- Stability: no saturation-guard trip in the run and feedback-to-source ratio over the last 2 s ≤ 10.0 dB (DESIGN.md §6).
- Limitation: LibriSpeech speech is 16 kHz upsampled (no content above 8 kHz); omni transducers; simulated rooms only.
- The full C29 gate (notch + NN vs notch only, C++ `FeedbackLoopSim`) runs on the local track.


## Outcome and diagnosis (Planner, 2026-10-18)

**Gate G2c: FAIL — stopped at step 19 823 after 15.85 of 40 chunk-hours** because
the phase-2 plateau rule fired and the SI-SDR loss-weight trial (DESIGN §9 B5-5)
did not lift it. This report evaluates `best_gate.pt` (step 17 856), the best
eval of the run; the export in this directory (`model.rtneural.json`,
`model.json`) is that checkpoint. It is published on `ckpt/s-v1` only, not under
`resources/models/` (that hand-back happens on PASS).

| Step | Phase | ASG median / p10 | SI-SDRi at −3 dB | ΔSTOI at −6 dB |
|---|---|---|---|---|
| 3 022 | 1 | +5.41 / +4.14 dB | +1.82 dB | — |
| 6 655 | 1 | +6.08 / +4.16 dB | +2.08 dB | +0.006 |
| 10 000 | 1 | +6.26 / +4.06 dB | +2.16 dB | +0.007 |
| 13 340 | 1 | +5.77 / +4.07 dB | +2.28 dB | +0.010 |
| 16 800 | 1 end | +6.08 / +4.02 dB | +2.30 dB | +0.010 |
| 17 312 | 2 | +5.87 / +3.98 dB | +2.40 dB | +0.009 |
| **17 856** | 2 | **+5.94 / +4.19 dB** | **+2.48 dB** | +0.010 |
| 18 382 | 2 | +6.12 / +4.30 dB | +2.39 dB | +0.009 |
| 19 085 | 2, SI-SDR weight 0.2 | +5.87 / +4.20 dB | +2.45 dB | +0.011 |
| 19 823 | 2, SI-SDR weight 0.2 | +5.91 / +4.33 dB | +2.41 dB | +0.008 |

Six of seven criteria pass with margin; SI-SDRi at −3 dB plateaued at
+2.4–2.5 dB against the fixed ≥ 3 dB threshold.

**Where the remaining error is** (8 eval scenes, `best_gate.pt`, median SI-SDR vs target):

| Signal | SI-SDR |
|---|---|
| Reverberant talker alone (late reverb is the only "error") | 15.43 dB |
| Model, open loop, on talker + noise | 15.48 dB |
| Model, open loop, on talker only | 16.26 dB |
| Identity loop at −3 dB (no processing) | 9.86 dB |
| Model inside the loop at −3 dB | 12.68 dB |

Noise and reverb are handled: open loop the model reaches the late-reverb
ceiling. Inside the loop it loses 2.8 dB, so the residual is feedback — a
delayed, room-filtered copy of the talker's own voice (−12.8 dB relative to
the talker at −3 dB, but coherent with it). A real-valued magnitude mask on
49 bins of 500 Hz cannot subtract a coherent copy that shares every
time-frequency cell with the direct voice; it can only attenuate cells, which
is why stability (ASG) is excellent while waveform fidelity (SI-SDRi) stalls.

**Proposals for DESIGN v2** (owner decision; the runbook routes this to a Planner session):

1. **Phase-aware output.** Replace the per-bin real mask with a complex mask on
   each Re/Im pair, or a short per-bin complex filter over the last 2–4 frames
   ("deep filtering"). Runs in C++ post-processing, keeps RTNeural to
   GRU → GRU → Dense; the Dense output grows from 49 to 98–392 values (S stays
   ≤ 250k parameters). Most likely to move SI-SDRi, because it can cancel a
   coherent delayed copy rather than only attenuate it.
2. **Reference input.** The device knows what it sends to the PA (its own
   output). Feeding the last output frame as a second input turns the task into
   learned feedback cancellation with a reference. Changes the model-format
   contract (two inputs per frame).
3. **Data realism.** 48 kHz wideband speech and more speakers (VCTK or
   train-clean-100 on the local M1 Pro; neither fits the cloud disk), measured
   RIRs, directional PA cabinets.
4. **Capacity.** Use the remaining S budget (192k → ≤ 250k parameters).
5. **Loss / schedule.** Tried and not sufficient on their own: earlier in-loop
   training (§9 B5-2/3) and a 4× SI-SDR weight (§9 B5-5).
