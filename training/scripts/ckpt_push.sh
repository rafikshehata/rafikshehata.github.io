#!/usr/bin/env bash
# Publish training/runs/<run>/ to the orphan branch ckpt/<run> as ONE commit
# (force-push), without touching the working tree or the current branch.
# Usage: ckpt_push.sh <run>      Remote: $CKPT_REMOTE (default origin)
set -euo pipefail
run="${1:?usage: ckpt_push.sh <run>}"
remote="${CKPT_REMOTE:-origin}"
here="$(cd "$(dirname "$0")/.." && pwd)"     # training/
repo="$(git -C "$here" rev-parse --show-toplevel)"
src="$here/runs/$run"
[ -d "$src" ] || { echo "ckpt_push: $src does not exist" >&2; exit 1; }

files=(latest.pt prev.pt best.pt best_gate.pt state.json metrics.jsonl chunks.jsonl babysit.jsonl evals_p2.jsonl REPORT.md report.json report_best_gate.json model.rtneural.json model.json)
limit=$((50 * 1024 * 1024))
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
export GIT_INDEX_FILE="$tmp/index"
n=0
for f in "${files[@]}"; do
  [ -f "$src/$f" ] || continue
  size=$(stat -c %s "$src/$f")
  if [ "$size" -gt "$limit" ]; then
    echo "ckpt_push: refusing $f ($size bytes > 50 MB)" >&2; exit 2
  fi
  blob=$(git -C "$repo" hash-object -w "$src/$f")
  git -C "$repo" update-index --add --cacheinfo "100644,$blob,$f"
  n=$((n + 1))
done
[ "$n" -gt 0 ] || { echo "ckpt_push: nothing to push in $src" >&2; exit 1; }
tree=$(git -C "$repo" write-tree)
step=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('step','?'))" "$src/state.json" 2>/dev/null || echo "?")
commit=$(git -C "$repo" -c user.name="${GIT_AUTHOR_NAME:-Claude}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@anthropic.com}" \
  commit-tree "$tree" -m "Stage3-cloud ckpt $run: step $step")
git -C "$repo" push --force "$remote" "$commit:refs/heads/ckpt/$run"
echo "ckpt_push: ckpt/$run -> $commit ($n files, step $step)"
