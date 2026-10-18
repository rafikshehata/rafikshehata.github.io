"""Download, verify and convert the s-v1 datasets into $RL_DATA_DIR (CLOUD-TRAINING.md §5).

    uv run python -m training.data.fetch --all          # everything (idempotent)
    uv run python -m training.data.fetch --only demand  # one group

Speech: LibriSpeech dev-clean (train) / test-clean (held-out), 16 kHz FLAC
resampled to 48 kHz mono FLAC (PCM_16). Noise: DEMAND 48 kHz, channel 1 only.
Archives are deleted after extraction. Every item gets a `.done` marker plus an
entry in $RL_DATA_DIR/manifest.json; training/DATASETS.md is regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import subprocess
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS = 48000
ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCES = {
    "dev-clean": dict(group="speech", split="train",
                      url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
                      md5="42e2234ba48799c1f50f24a7926300a1", licence="CC BY 4.0",
                      citation="Panayotov et al., LibriSpeech, ICASSP 2015 (openslr.org/12)"),
    "test-clean": dict(group="speech", split="held-out",
                       url="https://www.openslr.org/resources/12/test-clean.tar.gz",
                       md5="32fa31d27d2e1cad72775fee3f4849a9", licence="CC BY 4.0",
                       citation="Panayotov et al., LibriSpeech, ICASSP 2015 (openslr.org/12)"),
}
DEMAND_MD5 = {
    "OMEETING": "eae4c54c0e441df8375a979c0e9828f6",
    "OHALLWAY": "cb9227a75d2c1342de0b6548da4bbb1b",
    "PCAFETER": "b7b8b10f41b747cd0fd3849db13cebd2",
    "PRESTO": "6e9fef96d78e636029fd4c5929b9cece",
    "SPSQUARE": "f5fbdd5c1e03ab8c44c32ad96261ca26",
}
for env, md5 in DEMAND_MD5.items():
    SOURCES[f"demand-{env}"] = dict(
        group="demand", split="train: first 240 s / held-out: last 60 s",
        url=f"https://zenodo.org/api/records/1227121/files/{env}_48k.zip/content",
        md5=md5, licence="CC BY 4.0 (Zenodo record 1227121 metadata)",
        citation="Thiemann, Ito, Vincent, DEMAND, Zenodo 10.5281/zenodo.1227121")


def data_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("RL_DATA_DIR", pathlib.Path.home() / "RinglessData"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hashes(path: pathlib.Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # curl honours the agent proxy and its CA bundle; -C - resumes partial files.
    cmd = ["curl", "-fsSL", "--retry", "4", "--retry-delay", "2", "-C", "-", "-o", str(dest), url]
    for attempt in range(4):
        if subprocess.run(cmd).returncode == 0:
            return
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"download failed: {url} (if the host is blocked, the environment allowlist is missing it)")


def _convert_librispeech(args) -> int:
    rel, data = args
    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 16000, sr
    y = resample_poly(x, 3, 1).astype(np.float32)
    out = data_dir() / "librispeech" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.clip(y, -1, 1), FS, subtype="PCM_16", format="FLAC")
    return len(y)


def extract_librispeech(archive: pathlib.Path, name: str) -> dict:
    jobs, n_files, n_samples = [], 0, 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool, tarfile.open(archive, "r:gz") as tar:
        futures = []
        for m in tar:
            if not m.name.endswith(".flac"):
                continue
            parts = pathlib.PurePosixPath(m.name).parts       # LibriSpeech/<split>/<spk>/<chap>/<utt>.flac
            rel = pathlib.PurePosixPath(name, *parts[2:])
            futures.append(pool.submit(_convert_librispeech, (str(rel), tar.extractfile(m).read())))
        for f in futures:
            n_samples += f.result()
            n_files += 1
    return {"files": n_files, "hours_48k": round(n_samples / FS / 3600, 3)}


def extract_demand(archive: pathlib.Path, env: str) -> dict:
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.endswith("ch01.wav"))
        x, sr = sf.read(io.BytesIO(z.read(member)), dtype="float32")
    assert sr == FS, sr
    out = data_dir() / "demand" / f"{env}.flac"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, x, FS, subtype="PCM_16", format="FLAC")
    return {"files": 1, "audio_seconds": round(len(x) / FS, 2)}


def du(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fetch(name: str) -> dict:
    src, d = SOURCES[name], data_dir()
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    marker = d / ".done" / name
    if marker.exists() and name in manifest:
        print(f"fetch: {name} already done")
        return manifest[name]
    archive = d / "_dl" / pathlib.PurePosixPath(src["url"].replace("/content", "")).name
    t0 = time.time()
    print(f"fetch: downloading {name} ...", flush=True)
    download(src["url"], archive)
    sha, md5 = _hashes(archive)
    if md5 != src["md5"]:
        archive.unlink()
        raise RuntimeError(f"{name}: md5 mismatch {md5} != {src['md5']}")
    size = archive.stat().st_size
    print(f"fetch: extracting {name} ({size / 1e6:.0f} MB, md5 ok) ...", flush=True)
    if src["group"] == "speech":
        info = extract_librispeech(archive, name)
        out = d / "librispeech" / name
    else:
        env = name.split("-", 1)[1]
        info = extract_demand(archive, env)
        out = d / "demand" / f"{env}.flac"
    archive.unlink()
    entry = dict(url=src["url"], licence=src["licence"], citation=src["citation"], split=src["split"],
                 sha256=sha, md5=md5, download_bytes=size, on_disk_bytes=du(out),
                 path=str(out.relative_to(d)), elapsed_s=round(time.time() - t0, 1), **info)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(sha)
    print(f"fetch: {name} done in {entry["elapsed_s"]} s, {entry['on_disk_bytes'] / 1e6:.0f} MB on disk", flush=True)
    return entry


SOFTWARE = [
    ("torch", "BSD-3-Clause"), ("numpy", "BSD-3-Clause"), ("scipy", "BSD-3-Clause"),
    ("soundfile", "BSD-3-Clause"), ("pyroomacoustics", "MIT"), ("pystoi", "MIT"),
    ("pyyaml", "MIT"), ("tqdm", "MPL-2.0 AND MIT"), ("pytest", "MIT"), ("hatchling", "MIT"),
    ("RTNeural (runtime, not a Python dependency)", "BSD-3-Clause"),
]


def write_datasets_md() -> None:
    d = data_dir()
    manifest = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
    lines = ["# Datasets and software (s-v1)", "",
             "Generated by `training/data/fetch.py`; sizes measured on the cloud VM.",
             "Nothing here is committed except this table; data lives in `$RL_DATA_DIR`.", "",
             "| Item | URL | Licence | Split | SHA-256 (archive) | Download | On disk |",
             "|---|---|---|---|---|---|---|"]
    for name in SOURCES:
        e = manifest.get(name)
        if not e:
            lines.append(f"| {name} | {SOURCES[name]['url']} | {SOURCES[name]['licence']} | {SOURCES[name]['split']} | not fetched | | |")
            continue
        lines.append(f"| {name} | {e['url']} | {e['licence']} | {e['split']} | `{e['sha256']}` | "
                     f"{e['download_bytes'] / 1e6:.1f} MB | {e['on_disk_bytes'] / 1e6:.1f} MB |")
    lines += ["", "Citations: " + "; ".join(sorted({s['citation'] for s in SOURCES.values()})) + ".", "",
              "Limitation: LibriSpeech is 16 kHz audio upsampled to 48 kHz, so speech has no content above 8 kHz.", "",
              "RIRs are generated with pyroomacoustics (MIT) by `training/data/rir.py`; seeds are listed in DESIGN.md §1.1.", "",
              "## Software", "", "| Package | Licence |", "|---|---|"]
    lines += [f"| {p} | {l} |" for p, l in SOFTWARE]
    (ROOT / "DATASETS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", choices=["speech", "demand"])
    ap.add_argument("--name", choices=list(SOURCES), help="fetch a single item")
    a = ap.parse_args()
    names = [n for n in SOURCES if a.all or (a.only and SOURCES[n]["group"] == a.only) or n == a.name]
    for n in names:
        fetch(n)
    write_datasets_md()


if __name__ == "__main__":
    main()
