"""Review regressions: non-finite tensor views, sample-0 totals, zero ties in agreement, lens memory."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from neurotome.attribution import agreement, input_attribution
from neurotome.runs import RunManager
from neurotome.store import RunStore
from neurotome.tensors import tensor_view


def finite_stats(x: np.ndarray) -> dict[str, float]:
    f = x[np.isfinite(x)]
    return {"mean": f.mean(), "std": f.std(), "min": f.min(), "max": f.max()}


def test_tensor_view_non_finite_is_json_safe():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 4, 6, 6)).astype(np.float32)
    img[0, 0, 0, 0], img[0, 0, 4, 1], img[0, 1, 2, 2] = np.nan, np.inf, -np.inf
    seq = rng.standard_normal((1, 5, 8)).astype(np.float32)
    seq[0, 1, 2], seq[0, 3, 3] = np.nan, np.inf
    vec = rng.standard_normal((1, 9)).astype(np.float32)
    vec[0, 3], vec[0, 5] = np.nan, np.inf
    for a, kw, part in ((img, {"channel": 0}, img[0, 0]), (img, {"channel": 0, "norm": "global"}, img[0, 0]),
                        (seq, {}, seq[0]), (vec, {}, vec[0])):  # fmt: skip
        v = tensor_view(a, **kw)
        json.dumps(v, allow_nan=False)  # was NaN / inf, which the API sends as null: stats all "—"
        for k, ref in finite_stats(part).items():
            assert v["stats"][k] == pytest.approx(float(ref), rel=1e-5, abs=1e-6)
        assert sum(v["hist"]["counts"]) == int(np.isfinite(part).sum())
    assert tensor_view(img, channel=0)["peak"] == [4, 1]  # the +inf, never the NaN at (0, 0)
    heat = np.asarray(tensor_view(seq)["heatmap"])  # one inf zeroed it; one NaN left it unscaled
    assert np.abs(heat).max() == 1 and np.count_nonzero(heat) > heat.size // 2
    top = tensor_view(vec)["top"]
    assert top[0] == {"index": 5, "value": None} and 3 not in [t["index"] for t in top[:-1]]
    dead = tensor_view(np.full((1, 3, 4, 4), np.nan, np.float32))
    json.dumps(dead, allow_nan=False)
    assert dead["stats"] == {"mean": None, "std": None, "min": None, "max": None}


def test_tensor_route_serves_a_tensor_with_inf(tmp_path):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    class LogNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            y = torch.log(torch.relu(self.conv(x)))  # -inf wherever the ReLU is 0
            return self.fc(y.clamp(min=-5).mean((2, 3)))

    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    run = mgr.create_from_objects(LogNet(), torch.randn(1, 3, 8, 8), name="log")["id"]
    op = next(o for o in mgr.store.ir(run)["ops"] if o["type"] == "log")
    assert op["stats"]["inf"] > 0
    r = TestClient(create_app(mgr)).get(f"/api/runs/{run}/ops/{op['id']}/tensor")
    body = r.json()
    assert r.status_code == 200 and body["kind"] == "image"
    assert None not in body["stats"].values() and None not in body["pca"]["color"]


@pytest.mark.parametrize("method", ["ig", "gxa"])
def test_input_total_is_sample_0(method):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    x = torch.randn(2, 3, 2, 2)
    res = input_attribution(model, x, 1, method, 32)
    with torch.no_grad():  # linear: IG = g×x, and completeness holds per sample
        ref = float(model(x)[0, 1] - model(torch.zeros_like(x))[0, 1])
    assert res["total"] == pytest.approx(ref, abs=1e-5)  # was the sum over both samples


def test_agreement_ignores_zero_ties():
    a, b = np.zeros(100), np.zeros(100)
    a[[10, 20, 30]] = [3, 2, 1]
    b[[50, 60, 70]] = [3, 2, 1]
    r = agreement({"a": a, "b": b}, {}, "x")
    assert r["k"] == 10 and r["jaccard"][0][1] == 0 and r["overlap"][0][1] == 0  # was 0.54 / 0.7
    b[[10, 20, 30]] = [1, 2, 3]
    r = agreement({"a": a, "b": b}, {}, "x")
    assert r["jaccard"][0][1] == pytest.approx(0.5) and r["overlap"][0][1] == pytest.approx(0.3)
    same = agreement({"a": a, "b": a}, {}, "x")
    assert same["jaccard"][0][1] == 1 and same["overlap"][0][1] == pytest.approx(0.3)
    assert agreement({"a": np.zeros(5), "b": np.zeros(5)}, {}, "x")["jaccard"][0][1] == 1


def test_lens_projects_only_the_shown_positions():
    pytest.importorskip("transformers")
    from test_hf_v02 import tiny_gpt2

    from neurotome.lens import logit_lens

    torch.manual_seed(0)
    model = tiny_gpt2(vocab=64, positions=128).eval()
    model.config.use_cache = False
    x = torch.randint(0, 64, (1, 100))
    seen: list[int] = []
    handle = model.lm_head.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape[1]))
    try:
        res = logit_lens(model, x, None, None)
    finally:
        handle.remove()
    assert seen[0] == 100 and set(seen[1:]) == {64}  # the model's own pass, then one 64-row block per layer
    with torch.no_grad():
        p = torch.softmax(model(x).logits[0, 36:].float(), -1)
    assert torch.allclose(torch.tensor(res["target_prob"][-1]), p[:, res["target"]["id"]], atol=1e-4)
