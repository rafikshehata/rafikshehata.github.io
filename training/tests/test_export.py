"""B2: export checks (DESIGN.md section 10)."""
import json

import numpy as np
import pytest
import torch

from training.export import reference_infer, rtneural_json
from training.model.net import RinglessNet
from training.model.streaming import Streaming


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(1)
    net = RinglessNet((32, 24)).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
        net.dense.bias.normal_()
    out = rtneural_json.export(net, tmp_path_factory.mktemp("export"), name="test", run="test", step=3, config="x.yaml")
    return net, out


def test_reference_matches_torch(exported):
    net, out = exported
    x = (0.05 * torch.randn(1, 48 * 200)).double().float()
    s = Streaming(net); s.reset(1)
    masks = []
    with torch.no_grad():
        y_full, m_full = net(x, return_mask=True)
        y_stream = torch.cat([s.step(x[:, i:i + 48]) for i in range(0, x.shape[1], 48)], 1)[0].numpy()
    ref = reference_infer.ReferenceStage(out)
    y_ref = ref.process(x[0].numpy().astype(np.float64))
    m_ref = np.stack(ref.masks)
    assert np.abs(m_ref - m_full[0, :m_ref.shape[0]].numpy()).max() < 1e-4
    assert np.abs(y_ref - y_stream).max() < 1e-4
    assert np.abs(y_ref[48:] - y_full[0, :-48].numpy()).max() < 1e-4


def test_sidecar_validates(exported):
    _, out = exported
    side = rtneural_json.validate_sidecar(out / "model.json")
    assert side["latency_samples"] == 144 and side["stream_delay"] == 48
    side["window"] = 95
    bad = out / "bad.json"; bad.write_text(json.dumps(side))
    with pytest.raises(ValueError):
        rtneural_json.validate_sidecar(bad)


def test_rtneural_layer_list(exported):
    net, out = exported
    j = json.loads((out / "model.rtneural.json").read_text())
    assert j["in_shape"] == [None, None, 49]
    assert [l["type"] for l in j["layers"]] == ["gru", "gru", "dense"]
    assert j["layers"][2]["activation"] == "sigmoid"
    W, U, b = j["layers"][0]["weights"]
    assert np.shape(W) == (49, 96) and np.shape(U) == (32, 96) and np.shape(b) == (2, 96)
    K, bd = j["layers"][2]["weights"]
    assert np.shape(K) == (24, 49) and np.shape(bd) == (49,)
