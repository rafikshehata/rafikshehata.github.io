"""B4: eval harness checks (DESIGN.md section 10). Synthetic scenes from configs/tiny.yaml."""
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from training.data import loop_sim
from training.eval import asg as A

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenes():
    cfg = yaml.safe_load(open(ROOT / "configs" / "tiny.yaml"))
    return A.Scenes(cfg)


def test_identity_asg_zero(scenes):
    r = A.asg(scenes, loop_sim.Identity, loop_sim.Identity, iters=8)
    assert np.all(np.abs(r["asg"]) <= 0.3)
    assert abs(r["asg_median"]) <= 0.3


def test_fixed_gain_asg_six_db(scenes):
    r = A.asg(scenes, lambda: loop_sim.Gain(10 ** (-6 / 20)), loop_sim.Identity, iters=8)
    assert np.all(np.abs(r["asg"] - 6.0) <= 0.5), r["asg"]
    assert abs(r["asg_median"] - 6.0) <= 0.5


def test_random_checkpoint_report_fails():
    run = "_eval_random"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    p = subprocess.run([sys.executable, "-m", "training.eval.run", "--ckpt", "random", "--config", "configs/tiny.yaml",
                        "--run", run], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    text = (ROOT / "runs" / run / "REPORT.md").read_text()
    assert "## Gate G2c: FAIL" in text
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
