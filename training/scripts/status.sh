#!/usr/bin/env bash
# One-screen run status for the Babysitter. Usage: status.sh [run=$RL_RUN]
set -uo pipefail
run="${1:-${RL_RUN:-s-v1}}"
cd "$(dirname "$0")/.."
bash scripts/ckpt_pull.sh "$run" >/dev/null 2>&1 || true
RUN="$run" uv run --quiet python - <<'PY'
import json, os, re, shutil, pathlib
run = os.environ["RUN"]; d = pathlib.Path("runs") / run
def jl(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: return []
st = json.load(open(d / "state.json")) if (d / "state.json").exists() else {}
m = jl(d / "metrics.jsonl"); ch = jl(d / "chunks.jsonl")
val = [r for r in m if "val_sisdr" in r]
budget = None
try:
    txt = open("../ringlessmds/CLOUD-TRAINING.md").read()
    budget = float(re.search(r"budget_hours:\s*([\d.]+)", txt).group(1))
except Exception: pass
hours = sum(max(0, c["end"] - c["start"]) for c in ch) / 3600
free = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
print(f"run {run}")
print(f"step {st.get('step', 0)}  phase {st.get('phase', '-')}  last loss {m[-1].get('loss') if m else '-'}")
if val:
    v = val[-1]; print(f"last val: step {v['step']}  val_sisdr {v['val_sisdr']:.2f} dB  val_loss {v.get('val_loss', float('nan')):.4f}")
else: print("last val: -")
print(f"best val_sisdr {st.get('best_val_sisdr', '-')} at step {st.get('best_step', '-')}")
print(f"chunks {len(ch)}  last exit {ch[-1]['exit'] if ch else '-'}  chunk-hours {hours:.2f} / budget {budget}")
print(f"disk free {free:.1f} GB")
PY
