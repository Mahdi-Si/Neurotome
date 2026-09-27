"""Non-square attention: (Q, H, K) lag attention with exact zeros and masked rows, (H, Q, K) cross-attention,
and ordinary tensors that must not read as attention."""

from __future__ import annotations

import numpy as np
import pytest

from neurotome.tensors import _is_attention, tensor_view


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis, keepdims=True))
    return (e / e.sum(axis, keepdims=True)).astype(np.float32)


def lag_attention(T: int = 300, H: int = 4, L: int = 91, dead: int = 10) -> np.ndarray:
    """(1, T, H, L) entmax-like: sparse over lags (lag > t out of range), the first `dead` steps masked."""
    rng = np.random.default_rng(0)
    p = softmax(rng.normal(size=(T, H, L)) * 3)
    p[p < 0.02] = 0
    p *= np.arange(L)[None, None, :] <= np.arange(T)[:, None, None]  # lags before the start are out of range
    p[:, :, 0] += 1e-3  # every live row keeps lag 0
    p /= p.sum(-1, keepdims=True)
    p[:dead] = 0  # all-masked rows, nan_to_num'ed to zero
    return p[None].astype(np.float32)


def test_lag_attention_is_qhk():
    a = lag_attention()
    v = tensor_view(a, channel=2)
    assert v["kind"] == "attention" and v["layout"] == "qhk" and v["square"] is False
    assert (v["heads"], v["queries"], v["keys"], v["tokens"], v["head"]) == (4, 300, 91, 300, 2)
    assert [m["head"] for m in v["maps"]] == [0, 1, 2, 3]
    assert np.asarray(v["maps"][0]["map"]).shape == (64, 64)
    assert np.asarray(v["full"]["map"]).shape == (128, 91)
    assert 0 < v["full"]["max"] <= a[0, :, 2].max()  # max of the pooled rows
    assert len(v["key_profile"]) == 4
    for h, (s, prof) in enumerate(zip(v["summary"], v["key_profile"], strict=True)):
        assert len(prof) == 91 and sum(prof) == pytest.approx(1, abs=1e-3)
        assert set(s) == {"head", "entropy", "entropy_norm", "sparsity", "masked", "argmax", "argmax_weight"}
        live = a[0, 10:, h]
        assert s["masked"] == pytest.approx(10 / 300) and s["sparsity"] == pytest.approx((live == 0).mean())
        assert 0 < s["entropy_norm"] < 1 and s["entropy"] == pytest.approx(s["entropy_norm"] * np.log2(91))
        assert s["argmax"] == int(np.argmax(live.mean(0)))
        assert s["argmax_weight"] == pytest.approx(prof[s["argmax"]], abs=1e-4)


def test_heads_first_cross_attention():
    v = tensor_view(softmax(np.random.default_rng(1).normal(size=(1, 8, 16, 77))))
    assert (v["kind"], v["layout"], v["heads"], v["queries"], v["keys"]) == ("attention", "hqk", 8, 16, 77)
    assert "prev" not in v["summary"][0] and len(v["key_profile"][0]) == 77


def test_square_keeps_old_fields():
    a = softmax(np.random.default_rng(2).normal(size=(1, 4, 16, 16)) * 3)
    v = tensor_view(a)
    assert (v["kind"], v["layout"], v["square"]) == ("attention", "hqk", True)
    assert (v["heads"], v["tokens"], v["queries"], v["keys"]) == (4, 16, 16, 16)
    assert {"entropy", "prev", "first", "self", "pattern", "sparsity", "argmax"} <= set(v["summary"][0])
    assert v["summary"][0]["prev"] == pytest.approx(a[0, 0, np.arange(1, 16), np.arange(15)].mean())
    assert v["key_profile"][1] == pytest.approx(a[0, 1].mean(0), abs=1e-4)
    long = tensor_view(softmax(np.random.default_rng(3).normal(size=(1, 2, 300, 300))))
    assert len(long["key_profile"][0]) == 256  # pooled


def test_ordinary_tensors_are_not_attention():
    rng = np.random.default_rng(4)
    assert tensor_view(rng.normal(size=(1, 300, 4, 91)).astype(np.float32))["kind"] == "image"
    relu = np.maximum(rng.normal(size=(1, 16, 32, 32)), 0).astype(np.float32)
    assert tensor_view(relu)["kind"] == "image"
    assert tensor_view(relu / relu.max())["kind"] == "image"  # in [0, 1], rows do not sum to 1
    img = rng.normal(size=(1, 10, 8, 8))
    assert tensor_view(softmax(img, axis=1))["kind"] == "image"  # softmax over channels: rows along W are not
    assert tensor_view(1 / (1 + np.exp(-img)).astype(np.float32))["kind"] == "image"  # sigmoid
    assert tensor_view(np.zeros((1, 4, 20, 30), dtype=np.float32))["kind"] == "image"  # all rows "masked"
    half = lag_attention(dead=160)  # more than half the rows masked
    assert _is_attention(half[0]) is None
    assert _is_attention(softmax(rng.normal(size=(100, 100, 7)))) is None  # no axis small enough for heads
    assert _is_attention(np.ones((1, 1, 5), dtype=np.float32) / 5) is None  # a single row


def test_spatial_softmax_over_w_reads_as_heads():
    """A (C, H, W) softmax over W has every row a distribution: C <= H reads as C heads of H x W."""
    v = tensor_view(softmax(np.random.default_rng(5).normal(size=(1, 4, 10, 12))))
    assert (v["kind"], v["layout"], v["heads"], v["queries"], v["keys"]) == ("attention", "hqk", 4, 10, 12)
