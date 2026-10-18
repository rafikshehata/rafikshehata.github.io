#!/usr/bin/env bash
# One resumable training chunk (CLOUD-TRAINING.md section 8):
#   pull ckpt/<run> -> train <= <minutes> (hard cap +2 min) -> push ckpt/<run>.
# Runs in the foreground; launch it in the background and poll the log.
# Usage: chunk.sh <run> [minutes=25]   Config: configs/${RL_CONFIG:-<run prefix>}.yaml
set -uo pipefail
run="${1:?usage: chunk.sh <run> [minutes]}"; minutes="${2:-25}"
cd "$(dirname "$0")/.."                      # -> training/
cfg="configs/${RL_CONFIG:-${run%%-*}}.yaml"
[ -f "$cfg" ] || { echo "chunk: missing $cfg" >&2; exit 1; }
dir="runs/$run"; mkdir -p "$dir"

exec 9>"$dir/.chunk.lock"
flock -n 9 || { echo "chunk: another chunk of $run is running" >&2; exit 4; }

bash scripts/ckpt_pull.sh "$run" || { echo "chunk: ckpt_pull failed" >&2; exit 5; }

ls -1t "$dir"/chunk-*.log 2>/dev/null | tail -n +4 | xargs -r rm -f   # keep 3 newest logs
free_kb=$(df -Pk "$HOME" | awk 'NR==2{print $4}')
if [ "$free_kb" -lt $((3 * 1024 * 1024)) ]; then
  echo "chunk: less than 3 GB free ($((free_kb / 1024)) MB); not starting" >&2; exit 3
fi

log="$dir/chunk-$(date +%s).log"
start=$(date +%s)
timeout $((minutes * 60 + 120)) uv run python train.py --config "$cfg" --run "$run" --resume --max-minutes "$minutes" > "$log" 2>&1
code=$?
end=$(date +%s)
printf '{"start": %d, "end": %d, "minutes": %s, "exit": %d, "log": "%s"}\n' "$start" "$end" "$minutes" "$code" "$(basename "$log")" >> "$dir/chunks.jsonl"

push=0
bash scripts/ckpt_push.sh "$run" || push=$?
echo "chunk: train exit $code, ckpt_push exit $push, wall $((end - start)) s"
echo "chunk: last metrics: $(tail -n 1 "$dir/metrics.jsonl" 2>/dev/null)"
[ "$code" -eq 0 ] || exit "$code"
exit "$push"
