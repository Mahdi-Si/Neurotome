"""M2 backend: norm/cmap scaling, peak, slices, .npy export, diff, input preview, HTTP."""

from __future__ import annotations

import re
from io import BytesIO

import numpy as np
import pytest
import torch
from test_pipeline import Tiny

import neurotome.slicing
from neurotome.models import IMAGENET_MEAN, IMAGENET_STD
from neurotome.runs import RunManager
from neurotome.slicing import SliceError, SliceTooLarge, parse_slice, slice_npy, slice_table
from neurotome.store import PREVIEW, RunStore
from neurotome.tensors import input_preview, tensor_diff, tensor_view


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("m2") / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")
    return mgr, run, mgr.store.ir(run["id"])


@pytest.fixture(scope="module")
def client(captured):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    return TestClient(create_app(captured[0]))


RAMP = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
RAMPS = np.stack([(k + 1) * RAMP for k in range(3)])[None]  # (1, 3, 8, 8), channel k = (k+1) * ramp


def test_norm_channel_and_global():
    v = tensor_view(RAMPS)
    assert v["norm"] == "channel" and v["cmap"] == "seq" and v["scale"] is None
    for m in v["maps"]:
        assert np.max(m["map"]) == 1
        assert m["range"] == pytest.approx([0, m["channel"] + 1])
    assert v["dots_range"] == pytest.approx([0, v["channel"] + 1])

    g = tensor_view(RAMPS, norm="global")
    assert g["scale"] == pytest.approx([0, 3])
    assert [m["channel"] for m in g["maps"] if np.max(m["map"]) == 1] == [2]
    assert all(m["range"] == g["scale"] for m in g["maps"]) and g["dots_range"] == g["scale"]
    assert tensor_view(RAMPS, norm="global", lo=-10, hi=10)["scale"] == [-10, 10]


def test_default_output_matches_min_max():
    a = np.random.default_rng(1).normal(size=(1, 4, 30, 30)).astype(np.float32)
    v = tensor_view(a)
    for m in v["maps"]:
        grid = np.asarray(m["map"])
        assert grid.min() == 0 and grid.max() == 1


def test_diverging_keeps_zero():
    a = np.random.default_rng(0).normal(size=(1, 3, 8, 8)).astype(np.float32)
    a[0, :, 2, :] = 0
    v = tensor_view(a, cmap="div")
    assert v["cmap"] == "div"
    for m in v["maps"]:
        grid = np.asarray(m["map"])
        assert grid.min() >= -1 and grid.max() <= 1
        assert np.all(grid[2] == 0) and np.abs(grid).max() == 1
    dots = np.asarray(v["dots"])
    assert np.all(dots[2] == 0) and np.abs(dots).max() == 1


def test_sequence_shared_scale():
    a = np.full((1, 4, 6), 2.0, dtype=np.float32)
    own = tensor_view(a)
    assert own["scale"] is None and np.asarray(own["heatmap"]).max() == 1
    shared = tensor_view(a, norm="global", lo=-8, hi=4)
    assert shared["scale"] == [-8, 4] and np.allclose(shared["heatmap"], 0.25)


def test_peak_is_argmax_and_pixel_is_inspected():
    a = np.random.default_rng(2).normal(size=(1, 4, 8, 8)).astype(np.float32)
    v = tensor_view(a, pixel=(0, 0))
    assert v["vector"]["pixel"] == [0, 0] and v["profile"]["row"] == 0
    ch = a[0, v["channel"]]
    assert v["peak"] == list(np.unravel_index(int(np.argmax(ch)), ch.shape))


ARR = np.arange(1 * 64 * 56 * 56).reshape(1, 64, 56, 56)


def test_slice_table():
    t = slice_table(ARR, "[0, 12, 20:30, :]")
    assert t["shape"] == [10, 56] and t["axes"] == [2, 3] and not t["truncated"]
    assert t["indices"][0] == list(range(20, 30)) and t["key"] == "[0, 12, 20:30:1, 0:56:1]"
    assert t["values"] == ARR[0, 12, 20:30, :].ravel().tolist()
    assert slice_table(ARR, "[..., 0]")["full_shape"] == [1, 64, 56]
    assert slice_table(ARR, "[-1]")["full_shape"] == [64, 56, 56]
    back = slice_table(ARR, " [ 0 , 0 , 0 , ::-2 ] ")
    assert back["indices"][0] == list(range(55, -1, -2)) and back["values"] == ARR[0, 0, 0, ::-2].tolist()
    assert back["key"] == "[0, 0, 0, 55::-2]"
    assert parse_slice("0,1", ARR.shape) == parse_slice("[0, 1, :, :]", ARR.shape)
    big = slice_table(ARR, "[0]")
    assert big["shape"] == [1, 56, 56] and big["full_shape"] == [64, 56, 56] and big["truncated"]
    assert len(big["values"]) <= 4096 and big["indices"][0] == [0]
    scalar = slice_table(ARR, "0,0,0,5")
    assert scalar["shape"] == [] and scalar["axes"] == [] and scalar["values"] == [5]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[0,0,0,0,0]", "Too many indices: 5 for a 4-D tensor (1×64×56×56)."),
        ("[0, 99]", "Index 99 is out of range for axis 1 (size 64)."),
        ("[1:2:0]", "Slice step cannot be zero (axis 0)."),
        ("[a]", 'Cannot read "a" (item 1). Use integers, start:stop:step, ":" or "...".'),
        ("0,,1", 'Cannot read "" (item 2). Use integers, start:stop:step, ":" or "...".'),
        ("[..., ...]", 'Only one "..." is allowed.'),
        ("__import__('os')", "Cannot read \"__import__('os')\" (item 1)."),
        ("0" * 201, "Slice expression is too long (max 200 characters)."),
    ],
)
def test_slice_errors(text, message):
    with pytest.raises(SliceError, match=re.escape(message)):
        slice_table(ARR, text)


def test_slice_non_finite():
    a = np.array([[1.0, np.nan, np.inf, -np.inf, 1 / 3]], dtype=np.float32)
    assert slice_table(a, "")["values"] == [1.0, "nan", "inf", "-inf", 0.333333]


def test_slice_npy():
    out = np.load(BytesIO(slice_npy(ARR, "[0, 1:3]")))
    assert out.dtype == np.float32 and np.array_equal(out, ARR[0, 1:3].astype(np.float32))
    with pytest.raises(SliceTooLarge, match="Narrow the slice"):
        slice_npy(ARR, "", cap=10)


def test_tensor_diff():
    x = np.random.default_rng(3).normal(size=(2, 5, 7))
    same = tensor_diff(x, x)["diff"]
    assert same["max_abs"] == 0 and same["cosine"] == pytest.approx(1)
    d = tensor_diff(x, np.maximum(x, 0))["diff"]
    assert d["max_abs"] == pytest.approx(max(-x.min(), 0))
    assert sum(d["hist"]["counts"]) == x.size and len(d["hist"]["counts"]) == 24
    assert tensor_diff(x, x[0])["diff"] is None
    nan = tensor_diff(np.full(3, np.nan), np.zeros(3))["diff"]
    assert nan["stats"]["mean"] is None and nan["max_abs"] is None and nan["cosine"] is None
    assert tensor_diff(np.zeros(3), np.ones(3))["diff"]["rel_l2"] is None


def test_input_preview():
    p = np.random.default_rng(4).random((1, 3, 10, 12))
    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
    img = input_preview((p - mean) / std, (IMAGENET_MEAN, IMAGENET_STD))
    assert img.shape == (10, 12, 3) and img.dtype == np.uint8
    assert np.abs(img.astype(int) - np.round(p[0] * 255).transpose(1, 2, 0)).max() <= 1
    assert input_preview(np.zeros((1, 10))) is None
    assert input_preview(np.zeros((1, 3, 4, 40))) is None
    gray = input_preview(np.random.default_rng(5).normal(size=(1, 1, 9, 9)))
    assert gray.ndim == 2 and gray.min() == 0 and gray.max() == 255


def test_http_tensor_slice_npy(captured, client, monkeypatch):
    mgr, run, ir = captured
    conv = next(o for o in ir["ops"] if o["type"] == "conv2d")["id"]
    base = f"/api/runs/{run['id']}/ops/{conv}"
    t = client.get(f"{base}/tensor", params={"norm": "global", "cmap": "div"}).json()
    assert t["norm"] == "global" and t["cmap"] == "div" and len(t["scale"]) == 2
    assert client.get(f"{base}/tensor", params={"norm": "bad"}).status_code == 422
    assert client.get(f"{base}/slice", params={"s": "[0,0]"}).json()["op"] == conv
    bad = client.get(f"{base}/slice", params={"s": "[0,99]"})
    assert bad.status_code == 400 and "axis 1" in bad.json()["detail"]
    assert client.get(f"/api/runs/{run['id']}/ops/nope/slice").status_code == 404
    npy = client.get(f"{base}/slice.npy", params={"s": "[0, 1]"})
    assert npy.status_code == 200 and npy.headers["content-type"] == "application/octet-stream"
    assert "attachment" in npy.headers["content-disposition"]
    assert np.load(BytesIO(npy.content)).shape == (16, 16)
    monkeypatch.setattr(neurotome.slicing, "NPY_CAP", 10)
    assert client.get(f"{base}/slice.npy").status_code == 413


def test_http_diff(captured, client):
    _, run, ir = captured
    relu = next(o for o in ir["ops"] if o["type"] == "relu")
    r = client.get(f"/api/runs/{run['id']}/diff", params={"a": relu["id"], "b": relu["parents"][0]})
    body = r.json()
    assert r.status_code == 200 and body["same_shape"] and body["diff"]["max_abs"] >= 0
    assert body["a"] == relu["id"] and body["shape_a"] == relu["shape"]
    missing = client.get(f"/api/runs/{run['id']}/diff", params={"a": relu["id"], "b": "nope"})
    assert missing.status_code == 404 and missing.json()["detail"] == f"Op nope not found in {run['id']}"


def test_http_input_preview(captured, client):
    from PIL import Image

    mgr, run, _ = captured
    assert run["preview"] == {"width": 16, "height": 16}
    r = client.get(f"/api/runs/{run['id']}/input.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(r.content)).size == (16, 16)
    (mgr.store.run_dir(run["id"]) / PREVIEW).unlink()
    assert client.get(f"/api/runs/{run['id']}/input.png").status_code == 404
    flat = mgr.create_from_objects(torch.nn.Sequential(torch.nn.Linear(10, 4)), torch.randn(1, 10))
    assert flat["preview"] is None


def test_preview_failure_keeps_run(captured, monkeypatch):
    import neurotome.runs

    def boom(*_):
        raise ValueError("bad input")

    monkeypatch.setattr(neurotome.runs, "input_preview", boom)
    mgr = captured[0]
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))
    assert run["preview"] is None and mgr.store.exists(run["id"])


def test_spa_shell_revalidates(client):
    from neurotome.api import STATIC_DIR

    if not (STATIC_DIR / "index.html").exists():
        pytest.skip("frontend not built")
    assert client.get("/").headers["cache-control"] == "no-cache"
