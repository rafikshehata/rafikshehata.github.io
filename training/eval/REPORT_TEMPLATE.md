# REPORT — {run}

- checkpoint: `{ckpt}` (step {step}, phase {phase})
- config: `{config}` · git `{git_sha}` · generated {created}
- eval: {n_scenes} held-out scenes × {seconds} s, search [{search_lo}, {search_hi}] dB in {iters} steps, wall time {wall_s} s

## Gate G2c: {verdict}

| Metric | Value | Threshold | Result |
|---|---|---|---|
{gate_rows}

## Per-scene added stable gain (dB re identity MSG estimate)

| Scene | MSG identity | MSG model | ASG | Flags |
|---|---|---|---|---|
{scene_rows}

## Reported, not gated

| Metric | Value |
|---|---|
{extra_rows}

## Notes

- Stability: no saturation-guard trip in the run and feedback-to-source ratio over the last 2 s ≤ {fsr_max} dB (DESIGN.md §6).
- Limitation: LibriSpeech speech is 16 kHz upsampled (no content above 8 kHz); omni transducers; simulated rooms only.
- The full C29 gate (notch + NN vs notch only, C++ `FeedbackLoopSim`) runs on the local track.
{notes}
