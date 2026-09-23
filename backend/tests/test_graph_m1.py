"""M1 backend: fold overrides, NaN/Inf counts, dtype/device, search index, HTTP."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from test_pipeline import Tiny

from neurotome.capture import tensor_stats
from neurotome.graph import graph_view, search_index
from neurotome.runs import RunManager
from neurotome.store import RunStore


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("m1") / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")
    return mgr, run, mgr.store.ir(run["id"])


def ids(view, key="nodes"):
    return {x["id"] for x in view[key]}


def test_expand_opens_a_module(captured):
    ir = captured[2]
    base, opened = graph_view(ir, 0), graph_view(ir, 0, expand=["block:1"])
    assert "m:block:1" in ids(base) and "m:block:1" not in ids(opened)
    assert "g:block:1" in ids(opened, "groups")
    assert len(opened["nodes"]) > len(base["nodes"])


def test_expanded_single_child_keeps_its_box(tmp_path):
    from test_pipeline import Block

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Sequential(Block(3))

        def forward(self, x):
            return self.enc(x)

    mgr = RunManager(RunStore(tmp_path / "runs"))
    ir = mgr.store.ir(mgr.create_from_objects(Wrap(), torch.randn(1, 3, 8, 8), name="wrap")["id"])
    opened = graph_view(ir, 0, expand=["enc:1"])
    assert "m:enc.0:1" in ids(opened) and "g:enc:1" in ids(opened, "groups")


def test_collapse_folds_and_wins_over_expand(captured):
    ir = captured[2]
    both = graph_view(ir, 3, expand=["block:1"], collapse=["block:1"])
    for view in (graph_view(ir, 3, collapse=["block:1"]), both):
        assert "m:block:1" in ids(view) and "g:block:1" not in ids(view, "groups")


def test_no_overrides_is_unchanged(captured):
    ir = captured[2]
    for d in range(4):
        plain = json.dumps(graph_view(ir, d))
        assert json.dumps(graph_view(ir, d, None, (), ())) == plain
        assert json.dumps(graph_view(ir, d, expand=["nope:1"], collapse=["nope:2"])) == plain


def test_tensor_stats_counts_nonfinite():
    st = tensor_stats(np.array([1.0, np.nan, np.nan, np.inf, 0.0], dtype=np.float32))
    assert (st["nan"], st["inf"]) == (2, 1) and st["numel"] == 5
    st = tensor_stats(np.full(4, np.nan, dtype=np.float32))
    assert (st["nan"], st["inf"], st["mean"]) == (4, 0, None)


def test_node_nonfinite(captured):
    ir = copy.deepcopy(captured[2])
    conv = next(o for o in ir["ops"] if o["modules"][:1] == ["block:1"] and o["type"] == "conv2d")
    conv["stats"]["nan"] = 3
    node = next(n for n in graph_view(ir, 0)["nodes"] if n["id"] == "m:block:1")
    assert node["nonfinite"] >= 3
    for o in ir["ops"]:
        for k in ("nan", "inf"):
            (o["stats"] or {}).pop(k, None)
    assert all(n["nonfinite"] is None for n in graph_view(ir, 0)["nodes"])


def test_dtype_and_device(captured):
    ir = captured[2]
    view = graph_view(ir, 3)
    for x in view["nodes"] + view["edges"]:
        assert x["dtype"] == "torch.float32" and x["device"] == "cpu"
    for e in view["edges"]:
        assert e["target_dtype"] == "torch.float32" and e["target_device"] == "cpu"
    old = copy.deepcopy(ir)
    for o in old["ops"]:
        o.pop("device")
    view = graph_view(old, 3)
    assert all(x["device"] is None for x in view["nodes"] + view["edges"])
    assert len(search_index(old)) == len(search_index(ir))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_on_cuda(tmp_path):
    mgr = RunManager(RunStore(tmp_path / "runs"), device="cuda")
    run = mgr.create_from_objects(Tiny().cuda(), torch.randn(1, 3, 16, 16))
    view = graph_view(mgr.store.ir(run["id"]), 3)
    assert all(n["device"].startswith("cuda") for n in view["nodes"])


def test_search_index(captured):
    ir = captured[2]
    index = search_index(ir)
    assert [e["id"] for e in index[: len(ir["ops"])]] == [o["id"] for o in ir["ops"]]
    block = next(e for e in index if e["id"] == "m:block:1")
    assert block["kind"] == "module" and block["calls"] == [] and block["type"] == "Block"
    conv1 = next(e for e in index if e["path"] == "block.conv1")
    assert conv1["calls"][-1] == "block.conv1:1" and conv1["label"] == "conv1"


def test_http(captured):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    mgr, run, _ = captured
    c = TestClient(create_app(mgr))
    url = f"/api/runs/{run['id']}"
    base = c.get(f"{url}/graph", params={"depth": 0}).json()
    opened = c.get(f"{url}/graph", params={"depth": 0, "expand": "block:1,"}).json()
    assert base != opened and "m:block:1" not in ids(opened)
    r = c.get(f"{url}/index")
    assert r.status_code == 200 and isinstance(r.json(), list) and r.json()
    assert c.get("/api/runs/run_9999/index").status_code == 404
