#!/usr/bin/env bash
# Restore training/runs/<run>/ from branch ckpt/<run>. No-op if the branch does
# not exist. Keeps the local copy when it is at the same or a newer step.
# Usage: ckpt_pull.sh <run>      Remote: $CKPT_REMOTE (default origin)
set -euo pipefail
run="${1:?usage: ckpt_pull.sh <run>}"
remote="${CKPT_REMOTE:-origin}"
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
dst="$here/runs/$run"
rc=0; git -C "$repo" ls-remote --exit-code --heads "$remote" "ckpt/$run" >/dev/null 2>&1 || rc=$?
if [ "$rc" -eq 2 ]; then echo "ckpt_pull: no branch ckpt/$run on $remote (fresh run)"; exit 0; fi
if [ "$rc" -ne 0 ]; then echo "ckpt_pull: WARNING remote $remote unreachable (ls-remote exit $rc); using local runs/$run" >&2; exit 0; fi
git -C "$repo" fetch -q "$remote" "ckpt/$run"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git -C "$repo" archive --format=tar FETCH_HEAD | tar -xf - -C "$tmp"
step_of() { python3 -c "import json,sys;print(int(json.load(open(sys.argv[1])).get('step',-1)))" "$1" 2>/dev/null || echo -1; }
remote_step=$(step_of "$tmp/state.json"); local_step=$(step_of "$dst/state.json")
if [ "$local_step" -ge "$remote_step" ] && [ -f "$dst/state.json" ]; then
  echo "ckpt_pull: local step $local_step >= remote step $remote_step; keeping local"; exit 0
fi
mkdir -p "$dst"; cp -f "$tmp"/* "$dst"/
echo "ckpt_pull: restored ckpt/$run (step $remote_step) into $dst"
