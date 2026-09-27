"""v0.2 attention view: (H, T, T) softmax tensors open as `attention` with per-head maps and patterns."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neurotome.runs import RunManager
from neurotome.store import RunStore
from neurotome.tensors import _is_attention, tensor_view


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def tiny_gpt2(**kw):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=64,
                                  bos_token_id=0, eos_token_id=0, **kw)  # fmt: skip
    m = transformers.GPT2LMHeadModel(cfg)
    m.set_attn_implementation("eager")
    m.config.use_cache = False
    return m.eval()


def test_synthetic_softmax():
    a = softmax(np.random.default_rng(0).normal(size=(1, 4, 10, 10)) * 3)
    v = tensor_view(a)
    assert v["kind"] == "attention" and v["shape"] == [1, 4, 10, 10]
    assert (v["heads"], v["tokens"], v["head"]) == (4, 10, 0)
    assert [m["head"] for m in v["maps"]] == [0, 1, 2, 3]
    for h, m in enumerate(v["maps"]):
        assert np.asarray(m["map"]).shape == (10, 10) and m["max"] == pytest.approx(a[0, h].max())
        assert max(max(r) for r in m["map"]) == 1.0
    assert np.asarray(v["full"]["map"]).shape == (10, 10) and v["full"]["max"] == pytest.approx(a[0, 0].max())
    assert v["stats"]["min"] >= 0 and v["stats"]["max"] <= 1 and sum(v["hist"]["counts"]) == 100
    assert all({"head", "entropy", "prev", "first", "self", "pattern"} <= set(s) for s in v["summary"])
    assert tensor_view(a, channel=2)["head"] == 2 and tensor_view(a, channel=99)["head"] == 3
    assert tensor_view(a, channel=2)["full"]["max"] == pytest.approx(a[0, 2].max())


def test_pooling_caps():
    a = softmax(np.random.default_rng(1).normal(size=(1, 70, 200, 200)))
    v = tensor_view(a, channel=5)
    assert (v["heads"], v["tokens"], v["head"]) == (70, 200, 5)
    assert len(v["maps"]) == 64 and np.asarray(v["maps"][0]["map"]).shape == (64, 64)
    assert np.asarray(v["full"]["map"]).shape == (128, 128) and len(v["summary"]) == 70


def test_patterns_and_entropy():
    T = 8
    prev = np.eye(T, k=-1, dtype=np.float32)
    prev[0, 0] = 1
    uniform = np.full((T, T), 1 / T, dtype=np.float32)
    first = np.zeros((T, T), dtype=np.float32)
    first[:, 0] = 1
    s = tensor_view(np.stack([prev, uniform, first, np.eye(T, dtype=np.float32)])[None])["summary"]
    assert s[0]["prev"] == pytest.approx(1) and s[0]["pattern"] == "previous" and s[0]["entropy"] == 0
    assert s[1]["entropy"] == pytest.approx(math.log2(T)) and s[1]["pattern"] is None
    assert s[2]["first"] == 1 and s[2]["pattern"] == "first"
    assert s[3]["self"] == 1 and s[3]["pattern"] == "self"
    causal = np.tril(np.ones((T, T), dtype=np.float32))
    v = tensor_view((causal / causal.sum(-1, keepdims=True))[None, None])
    assert v["kind"] == "attention" and v["summary"][0]["first"] == pytest.approx(
        np.mean(1 / np.arange(2, T + 1))  # queries 1..T-1: query 0's weight on itself says nothing
    )


def test_bf16_softmax_is_attention():  # HF eager attention casts softmax back to bf16: rows sum to 1 +- ~3e-3
    torch.manual_seed(0)
    probs = torch.softmax(3 * torch.randn(1, 12, 32, 32), -1).to(torch.bfloat16).float().numpy()
    assert np.abs(probs.sum(-1) - 1).max() > 1e-3 and tensor_view(probs)["kind"] == "attention"


def test_non_attention_stays_image():
    rng = np.random.default_rng(2)
    a = softmax(rng.normal(size=(1, 4, 10, 10)))
    assert tensor_view(a * 2)["kind"] == "image"  # rows sum to 2
    assert tensor_view(a - 0.05)["kind"] == "image"  # negative
    assert tensor_view(rng.random((1, 8, 16, 16), dtype=np.float32))["kind"] == "image"
    assert tensor_view(softmax(rng.normal(size=(1, 4, 10, 12))) * 2)["kind"] == "image"  # rows sum to 2
    nan = a.copy()
    nan[0, 0, 0, 0] = np.nan
    assert not _is_attention(nan[0]) and tensor_view(nan)["kind"] == "image"
    assert tensor_view(np.ones((1, 3, 1, 1), dtype=np.float32))["kind"] != "attention"  # T = 1


def test_gpt2_softmax_op_through_api(tmp_path):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    run = mgr.create_from_objects(tiny_gpt2(), torch.randint(0, 64, (1, 7)))
    softmaxes = [o["id"] for o in mgr.store.ir(run["id"])["ops"] if o["type"] == "softmax"]
    assert len(softmaxes) == 2
    c = TestClient(create_app(mgr))
    v = c.get(f"/api/runs/{run['id']}/ops/{softmaxes[0]}/tensor", params={"channel": 1}).json()
    assert v["kind"] == "attention" and (v["heads"], v["tokens"], v["head"]) == (2, 7, 1)
    assert v["summary"][0]["first"] > 0 and np.asarray(v["full"]["map"]).shape == (7, 7)
