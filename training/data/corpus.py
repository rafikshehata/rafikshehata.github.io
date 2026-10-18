"""Speech and noise indices with speaker-disjoint splits (DESIGN.md §1.3).

Speech: train = LibriSpeech dev-clean, held-out = test-clean (48 kHz FLAC made by
fetch.py). Noise: DEMAND channel-1 files; train = first 240 s, held-out = rest.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import soundfile as sf

from training.data.fetch import data_dir

SPEECH_SPLITS = {"train": "dev-clean", "heldout": "test-clean"}
NOISE_ENVS = ["OMEETING", "OHALLWAY", "PCAFETER", "PRESTO", "SPSQUARE"]
FS = 48000


@dataclass(frozen=True)
class Utt:
    speaker: str
    path: str
    frames: int


def build_index(root: pathlib.Path) -> list[Utt]:
    """Index a LibriSpeech-style tree <root>/<speaker>/<chapter>/<utt>.flac (cached in index.json)."""
    cache = root / "index.json"
    if cache.exists():
        return [Utt(**u) for u in json.loads(cache.read_text())]
    utts = []
    for p in sorted(root.glob("*/*/*.flac")):
        utts.append(Utt(speaker=p.parts[-3], path=str(p.relative_to(root)), frames=sf.info(p).frames))
    if not utts:
        raise FileNotFoundError(f"no FLAC files under {root} — run `python -m training.data.fetch --all`")
    cache.write_text(json.dumps([u.__dict__ for u in utts]))
    return utts


def speakers(index: list[Utt]) -> set[str]:
    return {u.speaker for u in index}


def speech_index(split: str, base: pathlib.Path | None = None) -> tuple[pathlib.Path, list[Utt]]:
    base = base or data_dir() / "librispeech"
    root = base / SPEECH_SPLITS[split]
    return root, build_index(root)


def check_disjoint(base: pathlib.Path | None = None) -> tuple[int, int]:
    """Raise if any held-out speaker appears in the training index; return speaker counts."""
    _, tr = speech_index("train", base)
    _, ho = speech_index("heldout", base)
    overlap = speakers(tr) & speakers(ho)
    if overlap:
        raise AssertionError(f"held-out speakers in training split: {sorted(overlap)[:10]}")
    return len(speakers(tr)), len(speakers(ho))


def noise_segments(split: str, train_seconds: float = 240.0, base: pathlib.Path | None = None):
    """[(path, start, stop)] in samples for each DEMAND file; time-disjoint split (DESIGN §9 D4)."""
    base = base or data_dir() / "demand"
    out = []
    cut = int(train_seconds * FS)
    for env in NOISE_ENVS:
        p = base / f"{env}.flac"
        n = sf.info(p).frames
        out.append((str(p), 0, cut) if split == "train" else (str(p), cut, n))
    return out


if __name__ == "__main__":
    n_tr, n_ho = check_disjoint()
    print(f"speakers: train {n_tr}, held-out {n_ho}, overlap 0")
