"""Update the `run <run>:` line (and optionally others) of the STATE block in ringlessmds/CLOUD-TRAINING.md.

    python3 scripts/state_update.py s-v1 [--asg "median +x / p10 +y"] [--gate "PASS <sha>"] [--blocked "..."] [--note "..."]
Reads runs/<run>/state.json and chunks.jsonl. Prints the new line; the caller commits.
"""
import argparse
import datetime as dt
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT.parent / "ringlessmds" / "CLOUD-TRAINING.md"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--asg")
ap.add_argument("--gate")
ap.add_argument("--blocked")
ap.add_argument("--note")
a = ap.parse_args()
d = ROOT / "runs" / a.run
st = json.loads((d / "state.json").read_text())
chunks = [json.loads(l) for l in open(d / "chunks.jsonl")] if (d / "chunks.jsonl").exists() else []
hours = sum(c["end"] - c["start"] for c in chunks) / 3600
lv = (st.get("last_val") or {}).get("val_sisdr")
text = DOC.read_text()
old_asg = re.search(r"last ASG ([^\n]*)", text)
asg = a.asg or (old_asg.group(1) if old_asg else "—")
line = (f"- run {a.run}: chunks {len(chunks)} · step {st['step']} (phase {st['phase']}) · "
        f"val SI-SDR {lv:.2f} dB · best {st['best_val_sisdr']:.2f} dB @ {st['best_step']} · "
        f"chunk-hours {hours:.2f} · last ASG {asg}" if lv is not None else
        f"- run {a.run}: chunks {len(chunks)} · step {st['step']} · val SI-SDR — · best — · chunk-hours {hours:.2f} · last ASG {asg}")
text = re.sub(r"- run [^\n]*\n", line + "\n", text, count=1)
if a.gate:
    text = re.sub(r"- gate G2c: [^\n]*\n", f"- gate G2c: {a.gate}\n", text, count=1)
if a.blocked is not None:
    text = re.sub(r"- blocked: [^\n]*\n", f"- blocked: {a.blocked}\n", text, count=1)
note = a.note or f"chunk {len(chunks)} done"
text = re.sub(r"- last update: [^\n]*\n", f"- last update: {dt.date.today().isoformat()} — {note}\n", text, count=1)
DOC.write_text(text)
print(line)
