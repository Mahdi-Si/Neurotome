"""v0.2 NNsight backend: leaf-module calls as ops, same IR contract as TorchLens captures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_attention_v02 import tiny_gpt2
from test_pipeline import Tiny
from torch import nn

from neurotome import capture_nnsight as cn
from neurotome.graph import graph_view, module_site
from neurotome.models import to_logits
from neurotome.runs import RunManager
from neurotome.store import RunStore
from neurotome.tensors import tensor_view

pytest.importorskip("nnsight")

LEAVES = ["stem.0@1", "stem.1@1", "block.conv1@1", "block.relu@1", "block.conv2@1", "block.relu@2", "pool@1",
          "fc@1"]  # fmt: skip


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Res(nn.Module):
    """A leaf with two producers (input_1 and act@1): the merge falls back to parent shares."""

    def __init__(self):
        super().__init__()
        self.lin, self.act, self.add, self.head = nn.Linear(4, 4), nn.ReLU(), Add(), nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.add(x, self.act(self.lin(x))))


@pytest.fixture()
def store(tmp_path):
    torch.manual_seed(0)
    return RunStore(tmp_path / "runs")


def run(store: RunStore, model, x, **kw):
    run_id = store.new_run_id()
    store.create(run_id)
    return run_id, cn.capture(model, x, store, run_id, **kw)


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    torch.manual_seed(0)
    return RunManager(RunStore(tmp_path / "runs"))


def test_tiny_ops_are_leaf_calls(store):
    progress = []
    run_id, ir = run(store, Tiny(), torch.randn(1, 3, 16, 16), progress=lambda f, m: progress.append(m))
    ops = ir["ops"]
    assert [o["id"] for o in ops] == ["input_1", *LEAVES, "output_1"]
    assert ir["model"] | {"activation_bytes": 0} == {
        "class_name": "Tiny", "num_params": 1437, "num_ops": 8, "flops": 0, "activation_bytes": 0,
        "truncated": False, "budget_bytes": None, "capture_modules": None, "saved_ops": 8,
        "backend": "nnsight", "granularity": "module"}  # fmt: skip
    for o in ops:
        assert store.load_tensor(run_id, o["id"]).shape == tuple(o["shape"]) and o["saved"] and o["stats"]
    by = {o["id"]: o for o in ops}
    assert by["block.relu@2"]["modules"] == ["block:1", "block.relu:2"]
    assert by["stem.0@1"]["modules"] == ["stem:1", "stem.0:1"] and by["stem.0@1"]["module"] == "stem.0"
    assert by["block.conv1@1"]["type"] == "conv2d" and by["block.conv1@1"]["num_params"] == 8 * 8 * 9 + 8
    assert by["block.relu@2"]["parents"] == ["block.conv2@1"]  # the residual add is functional: previous leaf
    assert by["stem.0@1"]["parents"] == ["input_1"] and by["output_1"]["parents"] == ["fc@1"]
    assert all(c in by[p]["children"] for c, o in by.items() for p in o["parents"])
    assert ir["modules"]["block.relu"] == {"class_name": "ReLU", "num_params": 0, "depth": 2}
    assert ir["modules"][""]["class_name"] == "Tiny"
    assert progress[0] == "Trace" and progress[-1] == f"Save tensors {len(ops)}/{len(ops)}"
    assert module_site(ir, "block.conv1@1") == ("block.conv1", 1)
    assert module_site(ir, "block.relu@2") == ("block.relu", 2)
    assert [n["id"] for n in graph_view(ir, 0)["nodes"]] == ["input_1", "m:stem:1", "m:block:1", "pool@1",
                                                              "fc@1", "output_1"]  # fmt: skip
    deep = graph_view(ir, 3)
    assert {n["id"] for n in deep["nodes"]} == {o["id"] for o in ops}
    assert {g["id"] for g in deep["groups"]} == {"g:stem:1", "g:block:1"}


def test_gradients_keyed_by_op_ids(store):
    model, x = Tiny(), torch.randn(1, 3, 16, 16)
    _, ir = run(store, model, x)
    g = cn.op_gradients(model, x, None, to_logits)
    assert set(g["scores"]) == {o["id"] for o in ir["ops"]} and g["merges"] == {}
    assert g["target"] == int(model(x)[0].argmax()) and g["logit"] == pytest.approx(g["value"])
    assert len(g["channel_scores"]["block.conv1@1"]) == 8
    unit = cn.op_gradients(model, x, None, to_logits, {"op": "block.conv1@1", "channel": 3})
    assert unit["target"] is None and "block.conv1@1" in unit["scores"] and "fc@1" not in unit["scores"]
    with pytest.raises(ValueError, match="Unknown op"):
        cn.op_gradients(model, x, None, to_logits, {"op": "nope@1", "channel": 0})
    with pytest.raises(ValueError, match="out of range"):
        cn.op_gradients(model, x, None, to_logits, {"op": "fc@1", "channel": 9})


def test_merge_falls_back_to_parent_shares(store):
    model, x = Res(), torch.randn(1, 4)
    _, ir = run(store, model, x)
    assert next(o for o in ir["ops"] if o["id"] == "add@1")["parents"] == ["input_1", "act@1"]
    g = cn.op_gradients(model, x, 0, to_logits)
    view = graph_view(ir, 3, g["scores"], merges=g["merges"])
    into = [e for e in view["edges"] if e["target"] == "add@1"]
    assert len(into) == 2 and {e["share_method"] for e in into} == {"parent"}
    assert sum(e["share"] for e in into) == pytest.approx(1)


def test_no_grad(store):
    class Detached(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, x):
            return self.lin(x).detach()

    with pytest.raises(ValueError, match="no gradient"):
        cn.op_gradients(Detached(), torch.randn(1, 4), 0, to_logits)


def test_budget_and_selective(store):
    x = torch.randn(1, 3, 16, 16)
    budget = 3 * 16 * 16 * 4 + 5 * 4 + 2 * 8 * 16 * 16 * 4 + 100  # input, output and two conv-sized ops
    _, ir = run(store, Tiny(), x, max_bytes=budget)
    body = ir["ops"][1:-1]
    assert ir["model"]["truncated"] and ir["model"]["saved_ops"] == 2 and ir["ops"][-1]["saved"]
    assert [o["unsaved"] for o in body] == [None, None] + ["budget"] * 6 and all(o["stats"] for o in body)

    run_id, ir = run(store, Tiny(), x, modules=["block"])
    assert ir["model"]["capture_modules"] == ["block"] and ir["model"]["saved_ops"] == 4
    for o in ir["ops"]:
        inside = o["id"].startswith("block.") or o["is_input"] or o["is_output"]
        assert o["unsaved"] == (None if inside else "modules") and o["stats"] and o["saved"] == inside
    assert not store.tensor_path(run_id, "stem.0@1").exists()
    with pytest.raises(ValueError, match=r"Unknown module\(s\) nope; .*e\.g\. stem, block"):
        run(store, Tiny(), x, modules=["nope"])


def test_missing_and_patch(store, monkeypatch):
    monkeypatch.setitem(sys.modules, "nnsight", None)
    with pytest.raises(ValueError) as e:
        run(store, Tiny(), torch.randn(1, 3, 16, 16))
    assert str(e.value) == cn.MISSING
    with pytest.raises(ValueError) as e:
        cn.patch(Tiny(), torch.randn(1, 3, 16, 16), "fc@1", lambda t: t)
    assert str(e.value) == cn.PATCH


def test_trace_error(store):
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, x):
            raise RuntimeError("boom")

    from neurotome.errors import TraceError

    with pytest.raises(TraceError, match="NNsight could not trace forward\\(\\): RuntimeError: boom"):
        run(store, Broken(), torch.randn(1, 4))


def test_gpt2_attn_dropout_is_attention(store):
    # transformers 5 calls attn_dropout as a module only on GPT-2's reorder_and_upcast_attn path; the default
    # eager path applies dropout functionally, so no module holds the probabilities
    model, ids = tiny_gpt2(reorder_and_upcast_attn=True), torch.randint(0, 64, (1, 7))
    run_id, ir = run(store, model, ids)
    by = {o["id"]: o for o in ir["ops"]}
    assert by["input_1"]["dtype"] == "torch.int64" and by["output_1"]["shape"] == [1, 7, 64]
    assert by["transformer.h.0.attn.attn_dropout@1"]["modules"][-1] == "transformer.h.0.attn.attn_dropout:1"
    v = tensor_view(store.load_tensor(run_id, "transformer.h.0.attn.attn_dropout@1"))
    assert v["kind"] == "attention" and (v["heads"], v["tokens"]) == (2, 7)
    g = cn.op_gradients(model, ids, None, to_logits)
    assert "transformer.h.1.mlp.c_fc@1" in g["scores"] and np.isfinite(g["logit"])


# ---- through RunManager / the API (wired by runs.py, spec §6.3) -----------------------------------------


def test_run_manager_backend(mgr):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    meta = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), backend="nnsight")
    assert meta["model"]["backend"] == "nnsight" and meta["model"]["granularity"] == "module"
    res = mgr.attribution(meta["id"], "gxa", None, 8)
    ids = {o["id"] for o in mgr.store.ir(meta["id"])["ops"]}
    assert set(res["scores"]) <= ids and "block.relu@2" in res["scores"] and res["merges"] == {}
    c = TestClient(create_app(mgr))
    r = c.post(f"/api/runs/{meta['id']}/jobs", json={"kind": "patch", "op": "block.relu@2"})
    assert r.status_code == 400 and r.json()["detail"] == cn.PATCH
    graph = c.get(f"/api/runs/{meta['id']}/graph", params={"depth": 3, "method": "gxa"}).json()
    assert {n["output_op"] for n in graph["nodes"]} == ids


def test_auto_backend_and_missing(tmp_path, monkeypatch):
    x = torch.randn(1, 3, 16, 16)
    auto = RunManager(RunStore(tmp_path / "a"), nnsight_params=0).create_from_objects(Tiny(), x)
    assert auto["model"]["backend"] == "nnsight"
    assert (
        RunManager(RunStore(tmp_path / "b")).create_from_objects(Tiny(), x)["model"]["backend"] == "torchlens"
    )
    monkeypatch.setattr(cn, "available", lambda: False)
    with pytest.raises(ValueError) as e:
        RunManager(RunStore(tmp_path / "c")).create_from_objects(Tiny(), x, backend="nnsight")
    assert str(e.value) == cn.MISSING


def test_in_place_op_does_not_change_saved_values(store):
    # ReLU(inplace=True) overwrites bn's output after the hook: the saved values, stats and grad x act must be
    # bn's own (min < 0), as with TorchLens
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(inplace=True), nn.Flatten(),
                          nn.Linear(4 * 6 * 6, 3)).eval()  # fmt: skip
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        ref = model[1](model[0](x))
    run_id, ir = run(store, model, x)
    bn = next(o for o in ir["ops"] if o["id"] == "1@1")
    assert bn["stats"]["min"] == pytest.approx(float(ref.min())) and ref.min() < 0
    np.testing.assert_allclose(store.load_tensor(run_id, "1@1"), ref.numpy(), rtol=1e-6)
    g = cn.op_gradients(model, x, 0, to_logits)
    model[2].inplace = False
    act = model[1](model[0](x))
    act.retain_grad()
    model[4](model[3](model[2](act)))[0, 0].backward()
    assert g["scores"]["1@1"] == pytest.approx(float((act * act.grad).sum()), rel=1e-5)
