"""B1: simulator, dataset and corpus checks (DESIGN.md section 10). Synthetic sources: no downloads needed."""
import copy
import pathlib

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from scipy.signal import fftconvolve

from training.data import corpus, loop_sim, rir
from training.data.dataset import PairGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    c = yaml.safe_load(open(ROOT / "configs" / "tiny.yaml"))
    c["audio"]["seg_seconds"] = 2.0
    return c


@pytest.fixture(scope="module")
def gen(cfg):
    return PairGenerator(cfg, "train")


def test_same_seed_identical_pair(gen):
    a = gen.pair(123); b = gen.pair(123); c = gen.pair(124)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_shapes_and_dtype(gen):
    for seed in (1, 2, 3):
        mic, tgt, meta = gen.pair(seed)
        assert mic.shape == (96000,) and tgt.shape == (96000,)
        assert mic.dtype == np.float32 and tgt.dtype == np.float32
        assert np.isfinite(mic).all()


def _loop(gen, seed, g_rel, mute):
    cp = gen.components(seed, fixed={"mode": "id", "g_rel_db": g_rel, "headroom_db": 30.0, "snr_db": 20.0})
    with torch.no_grad():
        r = loop_sim.run_loop(cp["src"][None], cp["h_pa"][None], [cp["G"]], [cp["d"]], [cp["c"]],
                              loop_sim.Identity(), mute_on_trip=mute)
    return cp, r


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_identity_loop_howls_above_msg(gen, seed):
    cp, r = _loop(gen, seed, +3.0, mute=False)
    mic = r.mic[0].numpy(); q = 24000
    assert bool(r.tripped[0]), "guard must trip at +3 dB"
    assert loop_sim.energy_db(mic[-q:]) >= loop_sim.energy_db(mic[:q]) + 20
    _, r2 = _loop(gen, seed, +3.0, mute=True)
    t = int(r2.trip_at[0]); m2 = r2.mic[0].numpy()
    assert t > 0 and r2.tripped[0]
    assert np.all(r2.spk[0, t:].numpy() == 0), "PA must be muted after the trip"


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_identity_loop_bounded_below_msg(gen, seed):
    cp, r = _loop(gen, seed, -6.0, mute=False)
    assert not bool(r.tripped[0])
    q = 24000
    assert loop_sim.energy_db(r.mic[0, -q:].numpy()) <= loop_sim.energy_db(cp["src"][-q:]) + 20


def test_partitioned_conv_matches_fftconvolve():
    rng = np.random.default_rng(0)
    h = (rng.normal(size=(3, 5000)) * np.exp(-np.arange(5000) / 800)).astype(np.float32)
    x = rng.normal(size=(3, 96 * 50)).astype(np.float32)
    conv = loop_sim.PartitionedConv(torch.tensor(h))
    y = torch.cat([conv.step(torch.tensor(x[:, j * 96:(j + 1) * 96])) for j in range(50)], 1).numpy()
    ref = np.stack([fftconvolve(x[b], h[b])[:x.shape[1]] for b in range(3)])
    assert np.abs(y - ref).max() <= 1e-5 * np.abs(ref).max()


def test_msg_estimate_matches_brute_force():
    s = rir.make_scene(5, small=True)
    h = s["h_pa"].astype(np.float64)
    brute = -20 * np.log10(np.abs(np.fft.rfft(h, 64 * len(h))).max())
    assert abs(loop_sim.msg_db(h) - brute) < 0.01
    hn = h * s["pa_gain"]
    assert abs(loop_sim.msg_db(hn)) < 0.01, "normalised PA RIR must have MSG estimate 0 dB"


@pytest.mark.parametrize("n_clipped,expect", [(100, True), (99, False)])
def test_guard_threshold(n_clipped, expect):
    T = 96 * 20
    src = np.zeros((1, T), np.float32)
    pos = 600 + np.arange(n_clipped) * 4          # spread over 400 samples < 480 window
    src[0, pos] = 2.0                             # |G*p| = 2 >= c = 1
    r = loop_sim.run_loop(src, np.zeros((1, 96), np.float32), [1.0], [96], [1.0], loop_sim.Identity())
    assert bool(r.tripped[0]) == expect


def _fake_tree(base: pathlib.Path, split: str, speakers):
    for spk in speakers:
        p = base / split / spk / "100" / f"{spk}-100-0000.flac"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, np.zeros(4800, np.float32), 48000, format="FLAC")


def test_corpus_split_speaker_disjoint(tmp_path):
    _fake_tree(tmp_path, "dev-clean", ["11", "12"])
    _fake_tree(tmp_path, "test-clean", ["21", "22"])
    assert corpus.check_disjoint(tmp_path) == (2, 2)
    bad = tmp_path / "bad"
    _fake_tree(bad, "dev-clean", ["11", "12"])
    _fake_tree(bad, "test-clean", ["12", "22"])
    with pytest.raises(AssertionError):
        corpus.check_disjoint(bad)


def test_corpus_real_split_if_present():
    root = corpus.data_dir() / "librispeech"
    if not (root / "dev-clean").exists() or not (root / "test-clean").exists():
        pytest.skip("LibriSpeech not fetched on this VM")
    n_tr, n_ho = corpus.check_disjoint()
    assert n_tr >= 30 and n_ho >= 30


def test_targets_are_not_silent(cfg):
    """Long identity simulations must not crop past the utterance end (silent targets)."""
    c = copy.deepcopy(cfg); c["sim"]["id_sim_seconds"] = 4.0
    g = PairGenerator(c, "train")
    for seed in range(6):
        mic, tgt, meta = g.pair(300 + seed, fixed={"mode": "id"})
        assert loop_sim.energy_db(tgt) > loop_sim.energy_db(mic) - 50, meta
    cp = g.components(400, phase=2, T=4 * 48000)
    assert loop_sim.energy_db(cp["target"][-48000:]) > loop_sim.energy_db(cp["src"][-48000:]) - 30
