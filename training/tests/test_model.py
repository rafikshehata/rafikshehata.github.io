"""B2: model checks (DESIGN.md section 10)."""
import pathlib

import pytest
import torch
import yaml

from training.model import fb
from training.model.net import RinglessNet
from training.model.streaming import Streaming

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED = {"tiny": 5681, "s": 192561, "m": 661553}


def _cfg(name):
    return yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml"))


@pytest.mark.parametrize("name", ["tiny", "s"])
def test_streaming_equals_full_sequence(name):
    torch.manual_seed(0)
    net = RinglessNet.from_config(_cfg(name)).eval()
    with torch.no_grad():
        net.dense.bias.normal_()                      # non-trivial masks
        if net.A.requires_grad:
            net.A.add_(0.01 * torch.randn_like(net.A)); net.S.add_(0.01 * torch.randn_like(net.S))
    x = 0.05 * torch.randn(3, int(0.3 * 48000))
    with torch.no_grad():
        y = net(x)
    s = Streaming(net); s.reset(3)
    ys = torch.cat([s.step(x[:, i:i + 96]) for i in range(0, x.shape[1], 96)], 1)
    assert (ys[:, 48:] - y[:, :-48]).abs().max().item() < 1e-4


@pytest.mark.parametrize("name", ["tiny", "s", "m"])
def test_param_counts(name):
    net = RinglessNet.from_config(_cfg(name))
    n = net.param_count()
    assert abs(n - EXPECTED[name]) <= 0.1 * EXPECTED[name]
    assert n == EXPECTED[name] == _cfg(name)["model"]["expected_params"]
    if name == "s":
        assert n <= 250_000


def test_latency():
    net = RinglessNet.from_config(_cfg("s"))
    assert net.latency_samples == 144 and net.latency_samples / 48.0 == 3.0 <= 4.0


def test_initial_filterbank_reconstructs():
    net = RinglessNet((16, 16), learn_filterbank=False)
    with torch.no_grad():
        net.dense.bias.fill_(60.0)                    # mask -> 1
        x = 0.1 * torch.randn(2, 48 * 100)
        y = net(x)
        s = Streaming(net); s.reset(2)
        ys = torch.cat([s.step(x[:, i:i + 96]) for i in range(0, x.shape[1], 96)], 1)
    assert (y - x).abs().max() < 1e-4
    assert (ys[:, 48:] - x[:, :-48]).abs().max() < 1e-4


def test_bin_rows_partition():
    rows = fb.bin_rows(96)
    assert len(rows) == 49 and sorted(r for rr in rows for r in rr) == list(range(96))
