#!/usr/bin/env bash
# One-time (and idempotent) cloud setup: Python env + datasets + RIR bank.
# Safe to re-run; a second run on a ready VM finishes in well under 60 s.
set -euo pipefail
cd "$(dirname "$0")/.."                      # -> training/
export RL_DATA_DIR="${RL_DATA_DIR:-$HOME/RinglessData}"
mkdir -p "$RL_DATA_DIR"
t0=$(date +%s)

if [ -f uv.lock ]; then uv sync --python 3.12 --frozen; else uv sync --python 3.12; fi

# Data steps exist from B1 on; skip quietly before that.
if [ -f data/fetch.py ]; then uv run python -m training.data.fetch --all; fi
if [ -f data/rir.py ]; then uv run python -m training.data.rir --build; fi

echo "cloud_setup: done in $(( $(date +%s) - t0 )) s; data in $RL_DATA_DIR ($(du -sh "$RL_DATA_DIR" | cut -f1))"
