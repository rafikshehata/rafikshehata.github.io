"""B0: environment and scaffold checks (DESIGN.md section 10)."""
import importlib
import os
import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]          # training/
SCRIPTS = ["cloud_setup.sh", "chunk.sh", "ckpt_push.sh", "ckpt_pull.sh", "status.sh"]


def test_torch_is_cpu_build():
    import torch
    assert not torch.cuda.is_available()
    assert torch.version.cuda is None, f"CUDA build installed: {torch.version.cuda}"


@pytest.mark.parametrize("mod", ["numpy", "scipy", "soundfile", "pyroomacoustics", "pystoi", "yaml", "tqdm"])
def test_dependency_imports(mod):
    importlib.import_module(mod)


def test_package_importable_as_training():
    import training
    assert pathlib.Path(training.__file__).resolve().parent == ROOT


@pytest.mark.parametrize("name,params", [("tiny", 5681), ("s", 192561), ("m", 661553)])
def test_configs_load(name, params):
    cfg = yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))
    assert cfg["model"]["expected_params"] == params
    a = cfg["audio"]
    assert a["fs"] == 48000 and a["window"] + a["hop"] <= 192          # <= 4 ms at 48 kHz (D17)
    assert a["window"] <= 192 and a["hop"] <= 96


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_exist_and_parse(script):
    p = ROOT / "scripts" / script
    assert p.exists() and os.access(p, os.X_OK)
    subprocess.run(["bash", "-n", str(p)], check=True)


def test_gitignore_covers_heavy_dirs():
    text = (ROOT / ".gitignore").read_text()
    for entry in [".venv/", "runs/*", "data_cache/"]:
        assert entry in text
