"""End-to-end tests: capture -> graph view -> tensor view -> attribution -> HTTP API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neurotome.graph import graph_view, module_tree
from neurotome.runs import RunManager
from neurotome.store import RunStore
from neurotome.tensors import tensor_view


class Block(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.block = Block(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.block(self.stem(x))), 1))


@pytest.fixture()
def manager(tmp_path):
    torch.manual_seed(0)
    return RunManager(RunStore(tmp_path / "runs"))


@pytest.fixture()
def run(manager):
    return manager.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")


def test_capture_ir(manager, run):
    ir = manager.store.ir(run["id"])
    types = [o["type"] for o in ir["ops"]]
    assert types[0] == "input" and types[-1] == "output"
    assert "conv2d" in types and "linear" in types
    ids = {o["id"] for o in ir["ops"]}
    for o in ir["ops"]:
        assert set(o["parents"]) <= ids and set(o["children"]) <= ids
        if o["saved"]:
            arr = manager.store.load_tensor(run["id"], o["id"])
            assert list(arr.shape) == o["shape"]
    # the residual add has two parents
    assert any(len(o["parents"]) == 2 for o in ir["ops"])
    assert ir["modules"]["block"]["class_name"] == "Block"


def test_graph_view_collapses_by_depth(manager, run):
    ir = manager.store.ir(run["id"])
    shallow = graph_view(ir, depth=0)
    deep = graph_view(ir, depth=3)
    assert len(shallow["nodes"]) < len(deep["nodes"])
    kinds = {n["id"]: n["kind"] for n in shallow["nodes"]}
    assert "m:block:1" in kinds and kinds["m:block:1"] == "module"
    node_ids = {n["id"] for n in deep["nodes"]}
    for e in deep["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
    assert any(g["address"] == "block" for g in deep["groups"])
    assert [m["address"] for m in module_tree(ir)][:2] == ["stem", "stem.0"]


def test_tensor_views(manager, run):
    ir = manager.store.ir(run["id"])
    conv = next(o for o in ir["ops"] if o["type"] == "conv2d")
    v = tensor_view(manager.store.load_tensor(run["id"], conv["id"]), channel=3)
    assert v["kind"] == "image" and v["channel"] == 3 and v["channels"] == 8
    assert len(v["dots"]) == 16 and len(v["maps"]) == 8
    assert len(v["vector"]["values"]) == 8
    lin = next(o for o in ir["ops"] if o["type"] == "linear")
    assert tensor_view(manager.store.load_tensor(run["id"], lin["id"]))["kind"] == "vector"
    assert tensor_view(np.random.randn(1, 7, 12))["kind"] == "sequence"
    seq_first = tensor_view(np.random.randn(197, 1, 32))  # attention internals: (tokens, batch, dim)
    assert seq_first["kind"] == "sequence" and seq_first["tokens"] == 197


@pytest.mark.parametrize("method", ["gxa", "saliency", "ig"])
def test_attribution(manager, run, method):
    res = manager.attribution(run["id"], method, None, 8)
    assert 0 <= res["target"] < 5
    assert res["input"] and len(res["input"]["map"]) == 16
    assert res["scores"] and res["top_units"]
    if method == "ig":
        assert res["input"]["convergence_delta"] is not None
    view = graph_view(manager.store.ir(run["id"]), 3, res["scores"])
    assert view["path"][0] == next(n["id"] for n in view["nodes"] if n["is_output"])
    merge = [e for e in view["edges"] if "share" in e]
    assert merge and all(0 <= e["share"] <= 1 for e in merge)


def test_http_api(manager, run):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    c = TestClient(create_app(manager))
    assert c.get("/api/health").json()["ok"]
    assert c.get("/api/runs").json()[0]["id"] == run["id"]
    g = c.get(f"/api/runs/{run['id']}/graph", params={"depth": 2}).json()
    assert g["nodes"] and g["edges"]
    node = next(n for n in g["nodes"] if n["type"] == "conv2d")
    t = c.get(f"/api/runs/{run['id']}/ops/{node['output_op']}/tensor").json()
    assert t["kind"] == "image"
    a = c.post(f"/api/runs/{run['id']}/attribution", json={"method": "gxa"}).json()
    assert "scores" in a
    g2 = c.get(f"/api/runs/{run['id']}/graph", params={"depth": 2, "method": "gxa"}).json()
    assert "path" in g2
    assert c.get("/api/runs/run_9999").status_code == 404


def test_torchvision_resnet18(manager):
    meta = manager.create_from_spec("torchvision:resnet18", None, {"kind": "random", "shape": [1, 3, 64, 64]})
    ir = manager.store.ir(meta["id"])
    assert ir["model"]["num_ops"] == 69
    view = graph_view(ir, depth=2)
    assert any(g["address"] == "layer1.0" for g in view["groups"])
    assert sum(1 for n in view["nodes"] if n["type"] == "add") == 8


def test_stats_on_constant_tensors():
    from neurotome.capture import tensor_stats

    for arr in (
        np.full((1, 4, 3, 3), 2.5, dtype=np.float32),
        np.full(8, 1e8, dtype=np.float32) + 1e-3,
        np.array([100.0, 100.00001] * 4, dtype=np.float32),
    ):
        st = tensor_stats(arr)
        assert sum(st["hist"]) == arr.size
        assert tensor_view(arr)["hist"]["counts"]


def test_live_models_are_lru_bounded(tmp_path):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    mgr = RunManager(RunStore(tmp_path / "runs"), max_live_models=2)
    ids = [mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))["id"] for _ in range(3)]
    assert list(mgr._live) == ids[1:]
    with pytest.raises(RuntimeError, match="no longer loaded"):
        mgr.attribution(ids[0], "gxa", None, 8)  # evicted in-process model cannot come back
    mgr.attribution(ids[1], "gxa", None, 8)  # touch -> most recently used
    assert list(mgr._live) == [ids[2], ids[1]]
    assert TestClient(create_app(mgr)).delete(f"/api/runs/{ids[1]}").json()["deleted"]
    assert list(mgr._live) == [ids[2]] and not mgr.store.exists(ids[1])


def test_evicted_spec_run_reloads(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    mgr = RunManager(RunStore(tmp_path / "runs"), max_live_models=1)
    first = mgr.create_from_spec("py:test_pipeline:Tiny", None, {"kind": "random", "shape": [1, 3, 16, 16]})
    mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))
    assert first["id"] not in mgr._live
    assert mgr.attribution(first["id"], "gxa", None, 8)["scores"]
    assert list(mgr._live) == [first["id"]]


def test_max_live_models_setting(monkeypatch):
    from neurotome.config import get_settings

    monkeypatch.delenv("NEUROTOME_MAX_LIVE_MODELS", raising=False)
    assert get_settings().max_live_models == 2
    monkeypatch.setenv("NEUROTOME_MAX_LIVE_MODELS", "5")
    assert get_settings().max_live_models == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("method", ["gxa", "ig"])
def test_capture_and_attribution_on_cuda(tmp_path, method):
    mgr = RunManager(RunStore(tmp_path / "runs"), device="cuda")
    run = mgr.create_from_objects(Tiny().cuda(), torch.randn(1, 3, 16, 16))
    assert mgr.store.ir(run["id"])["ops"]
    res = mgr.attribution(run["id"], method, None, 8)
    assert res["scores"] and len(res["input"]["map"]) == 16
