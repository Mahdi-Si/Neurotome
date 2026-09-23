"""M3 backend: merge shares, unit objectives, layer / neuron methods, agreement, IG progress."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_pipeline import Tiny
from torch import nn

from neurotome.attribution import _avg_ranks, agreement, channel_conductance, objective_forward
from neurotome.capture import op_gradients
from neurotome.graph import graph_view, module_site
from neurotome.jobs import TERMINAL
from neurotome.models import to_logits
from neurotome.runs import RunManager
from neurotome.store import RunStore

UNIT = {"op": "relu_2_4", "channel": 3}  # Tiny: block.relu, first call


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("m3") / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")
    return mgr, run["id"], mgr.store.ir(run["id"])


@pytest.fixture(scope="module")
def resnet(tmp_path_factory):
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("m3r") / "runs"))
    run = mgr.create_from_spec("torchvision:resnet18", None, {"kind": "random", "shape": [1, 3, 64, 64]})
    return mgr, run["id"], mgr.store.ir(run["id"])


def finish(mgr, job, timeout=120.0):
    v = -1
    while True:
        data, v = mgr.jobs.wait(job.id, v, timeout)
        if data["status"] in TERMINAL:
            return data


def incoming(view, node):
    return [e for e in view["edges"] if e["target"] == node]


def test_merges_on_tiny(captured):
    mgr, run, ir = captured
    live = mgr.live(run)
    res = op_gradients(live.loaded.model, live.inputs, None, to_logits)
    add = res["merges"]["add_1_6"]
    assert set(add) == {"relu_1_2", "conv2d_3_5"}
    assert sum(add.values()) == pytest.approx(res["scores"]["add_1_6"], rel=1e-4)
    view = graph_view(ir, 3, res["scores"], merges=res["merges"])
    es = incoming(view, "add_1_6")
    total = sum(abs(v) for v in add.values())
    assert len(es) == 2 and all(e["share_method"] == "merge" for e in es)
    for e in es:
        assert e["share"] == pytest.approx(abs(add[e["source"]]) / total)
    legacy = graph_view(ir, 3, res["scores"])
    mags = {e["source"]: abs(res["scores"][e["source"]]) for e in incoming(legacy, "add_1_6")}
    for e in incoming(legacy, "add_1_6"):
        assert e["share_method"] == "parent"
        assert e["share"] == pytest.approx(mags[e["source"]] / sum(mags.values()))
    single = [e for e in legacy["edges"] if len(incoming(legacy, e["target"])) == 1]
    assert single and all(e["share"] == 1.0 and "share_method" not in e for e in single)


def test_merges_on_resnet18(resnet):
    mgr, run, ir = resnet
    res = mgr.attribution(run, "gxa", None, 8)
    iadds = [o["id"] for o in ir["ops"] if o["type"] == "iadd"]
    assert len(iadds) == 8
    for op in iadds:
        assert sum(res["merges"][op].values()) == pytest.approx(res["scores"][op], rel=1e-3)
    merged = graph_view(ir, 3, res["scores"], merges=res["merges"])
    legacy = graph_view(ir, 3, res["scores"])
    old = {(e["source"], e["target"]): e["share"] for e in legacy["edges"]}
    shared = [e for e in merged["edges"] if e.get("share_method")]
    diffs = [abs(e["share"] - old[e["source"], e["target"]]) for e in shared]
    assert all(e["share_method"] == "merge" for e in merged["edges"] if e["target"] in iadds)
    assert max(diffs) > 0.05


class Cat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 4, 3, padding=1)
        self.conv_b = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(torch.cat([self.conv_a(x), self.conv_b(x)], 1).mean((2, 3)))


def test_cat_merge_shares(tmp_path):
    torch.manual_seed(0)
    model = Cat()
    with torch.no_grad():
        model.conv_b.weight.zero_()
        model.conv_b.bias.zero_()
    mgr = RunManager(RunStore(tmp_path / "runs"))
    run = mgr.create_from_objects(model, torch.randn(1, 3, 8, 8))["id"]
    ir = mgr.store.ir(run)
    res = mgr.attribution(run, "gxa", None, 8)
    view = graph_view(ir, 3, res["scores"], merges=res["merges"])
    cat = next(o["id"] for o in ir["ops"] if o["type"] == "cat")
    shares = {view_path(view, e["source"]): e for e in incoming(view, cat)}
    assert shares["conv_b"]["share"] == 0 and shares["conv_a"]["share"] == pytest.approx(1.0)
    assert shares["conv_a"]["share_method"] == "merge"


def view_path(view, nid):
    return next(n["path"] for n in view["nodes"] if n["id"] == nid)


def test_unit_objective(captured):
    mgr, run, ir = captured
    res = mgr.attribution(run, "gxa", None, 8, objective=UNIT)
    saved = mgr.store.load_tensor(run, UNIT["op"])
    assert res["value"] == pytest.approx(float(np.asarray(saved)[0, 3].sum()), abs=1e-4)
    assert res["prob"] is None and res["target"] is None and res["logit"] is None
    assert res["objective"] == {"kind": "unit", "op": "relu_2_4", "channel": 3}
    assert res["objective_label"] == "block.relu · ch 3 · Σ spatial"
    for later in ("conv2d_3_5", "add_1_6", "relu_3_7", "linear_1_10"):
        assert later not in res["scores"]
    assert "conv2d_2_3" in res["scores"]
    files = [p.name for p in (mgr.store.run_dir(run) / "attribution").iterdir()]
    assert any("u_relu_2_4_c3" in f for f in files) and "channels_u_relu_2_4_c3.json" in files
    view = graph_view(ir, 3, res["scores"], merges=res["merges"], start=UNIT["op"])
    assert view["path"][0] == "relu_2_4"
    ig = mgr.attribution(run, "ig", None, 8, objective=UNIT)
    assert ig["input"]["convergence_delta"] is not None
    with pytest.raises(ValueError, match="not the output of a module"):
        mgr.attribution(run, "gxa", None, 8, objective={"op": "add_1_6", "channel": 0})
    with pytest.raises(ValueError, match="Unknown op"):
        mgr.attribution(run, "gxa", None, 8, objective={"op": "nope", "channel": 0})


def test_logit_result_schema(captured):
    mgr, run, _ = captured
    res = mgr.attribution(run, "gxa", 2, 8)
    assert res["objective"] == {"kind": "logit", "index": 2} and res["objective_label"] == "logit[2]"
    assert res["value"] == res["logit"] and res["merge_method"] and "channel_scores" not in res
    name = "attribution/saliency_top_8.json"  # an M2 cache file (no merges) is stale
    mgr.store.write_json(run, name, {"scores": {}})
    assert "merges" in mgr.attribution(run, "saliency", None, 8)


def test_layer_methods(captured, resnet):
    mgr, run, ir = captured
    live = mgr.live(run)
    model, x = live.loaded.model, live.inputs
    res = mgr.layer(run, "layer_ig", "conv2d_2_3", None, 8)
    assert len(res["channels"]) == 8 and res["path"] == "block.conv1" and res["method_label"]
    t = int(to_logits(model(x))[0].argmax())
    with torch.no_grad():
        diff = float(model(x)[0, t] - model(torch.zeros_like(x))[0, t])
    assert abs(res["total"] - diff) <= res["convergence_delta"] + 1e-4
    assert res["objective_label"] == f"logit[{t}]" and len(res["top"]) == 8
    assert len(mgr.layer(run, "layer_cond", "conv2d_2_3", None, 8)["channels"]) == 8
    unit = mgr.layer(run, "layer_ig", "conv2d_2_3", None, 8, objective=UNIT)
    assert unit["objective_label"] == "block.relu · ch 3 · Σ spatial"
    with pytest.raises(ValueError, match="called 2 times"):
        mgr.layer(run, "layer_ig", "relu_2_4", None, 8)
    assert module_site(ir, "relu_3_7") == ("block.relu", 2)
    folded = next(n for n in graph_view(ir, 0)["nodes"] if n["id"] == "m:block:1")
    assert module_site(ir, folded["output_op"]) == ("block.relu", 2)  # innermost call with that output
    assert mgr.layer(run, "layer_ig", folded["output_op"], None, 8)["path"] == "block"  # called once
    assert module_site(ir, "add_1_6") is None
    rir = resnet[2]
    bn = next(o for o in rir["ops"] if o["type"] == "batchnorm")
    assert bn["module"] is None and module_site(rir, bn["id"]) == ("bn1", 1)


def test_neuron_conductance(captured):
    from captum.attr import NeuronConductance

    mgr, run, _ = captured
    live = mgr.live(run)
    model, x = live.loaded.model, live.inputs
    fwd, _ = objective_forward(model, None, None)
    t, c = 1, 3
    ours = channel_conductance(model, x, model.fc, c, fwd, t, 8)
    ref = NeuronConductance(fwd, model.fc).attribute(x, neuron_selector=c, target=t, n_steps=8,
                                                      method="riemann_middle")  # fmt: skip
    assert torch.allclose(ours, ref, atol=1e-5)
    res = mgr.neuron(run, "linear_1_10", c, t, 8)
    assert len(res["input"]["map"]) == 16 and res["input"]["convergence_delta"] is None
    assert res["method"] == "neuron_cond" and res["path"] == "fc" and res["channel"] == c
    with pytest.raises(ValueError, match="out of range"):
        mgr.neuron(run, "linear_1_10", 9, t, 8)


def test_agreement_maths():
    v = np.array([1.0, 5, 3, 2, 4, 0.5, 7, 6, 9, 8])
    same = agreement({"a": v, "b": v}, {}, "x")
    for m in ("jaccard", "overlap", "spearman"):
        assert same[m] == [[1.0, 1.0], [1.0, 1.0]]
    assert same["n"] == 10 and same["k"] == 1 and same["labels"] == ["a", "b"]
    assert agreement({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}, {}, "x")["spearman"][0][1] == pytest.approx(-1)
    e = np.eye(10)
    disjoint = agreement({"a": e[0], "b": e[1]}, {}, "x")
    assert disjoint["jaccard"][0][1] == 0 and disjoint["overlap"][0][1] == 0
    const = agreement({"a": [2, 2, 2, 2], "b": [1, 2, 3, 4]}, {}, "x")
    assert const["spearman"][0][1] is None and const["spearman"][1][0] is None
    assert _avg_ranks(np.array([1.0, 2, 2, 3])).tolist() == [0, 1.5, 1.5, 3]
    ties = agreement({"a": [1, 2, 2, 3], "b": [1, 2, 3, 4]}, {}, "x")
    assert ties["spearman"][0][1] == pytest.approx(3 / np.sqrt(10))
    assert agreement({"a": [-4, 1], "b": [4, 1]}, {}, "x")["spearman"][0][1] == pytest.approx(1)  # |value|


def test_agreement_job(captured):
    mgr, run, _ = captured
    res = mgr.agreement(run, None, 8, op="conv2d_2_3")
    block = res["input"]
    assert block["methods"] == ["gxa", "saliency", "ig"] and len(block["jaccard"]) == 3
    for m in ("jaccard", "overlap", "spearman"):
        mat = np.array(block[m], dtype=float)
        assert np.allclose(mat, mat.T) and np.allclose(np.diag(mat), 1)
    units = res["units"]
    assert len(units["methods"]) >= 2 and units["n"] == 8 and units["op"] == "conv2d_2_3"
    assert res["units_error"] is None and res["objective_label"].startswith("logit[")
    bad = mgr.agreement(run, None, 8, op="relu_2_4")
    assert "called 2 times" in bad["units_error"]


def test_ig_progress(captured):
    mgr, run, _ = captured
    seen = []
    mgr.attribution(run, "ig", None, 32, progress=lambda f, m: seen.append((f, m)))
    msgs = [m for _, m in seen]
    assert msgs[0] == "Node gradients" and "IG batch 1/2" in msgs and "IG batch 2/2" in msgs
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and fracs[-1] <= 1


def hooked(model, module, x, value):
    handle = module.register_forward_hook(lambda _m, _i, out: value(out))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        handle.remove()


def test_random_weights_reload_identically():
    """Attribution and interventions reload the model from its spec; random-init weights match the capture."""
    from neurotome.models import load_model

    a, b = (load_model("torchvision:resnet18").model.conv1.weight for _ in range(2))
    assert torch.equal(a, b)


def test_layer_after_unit_objective(captured):
    mgr, run, _ = captured
    with pytest.raises(ValueError, match="does not feed the objective"):
        mgr.layer(run, "layer_ig", "conv2d_3_5", None, 8, objective=UNIT)  # block.conv2 runs after block.relu
    with pytest.raises(ValueError, match="does not feed the objective"):
        mgr.neuron(run, "conv2d_3_5", 0, None, 8, objective=UNIT)


def test_inplace_consumers_resnet18(resnet):
    """bn1 feeds ReLU(inplace) and layer1.0.bn2 feeds `out += identity`: methods explain its own value."""
    mgr, run, ir = resnet
    live = mgr.live(run)
    model, x = live.loaded.model, live.inputs
    for path, c in (("bn1", 0), ("layer1.0.bn2", 3)):
        op = next(o["id"] for o in ir["ops"] if o["modules"][-1:] == [f"{path}:1"])
        fwd, _ = objective_forward(model, (path, 1), c)
        stored = float(np.asarray(mgr.store.load_tensor(run, op))[0, c].sum())
        assert float(fwd(x)) == pytest.approx(stored, rel=1e-4, abs=1e-3)
        for m in ("layer_ig", "layer_cond"):
            assert len(mgr.layer(run, m, op, None, 4)["channels"]) == 64
        assert mgr.neuron(run, op, c, None, 4)["input"]["map"]
    unit = {"op": op, "channel": 3}  # layer IG on layer1.0.bn1 explains this unit up to its convergence delta
    bn1 = next(o["id"] for o in ir["ops"] if o["modules"][-1:] == ["layer1.0.bn1:1"])
    res = mgr.layer(run, "layer_ig", bn1, None, 16, objective=unit)
    fwd, _ = objective_forward(model, ("layer1.0.bn2", 1), 3)
    assert abs(res["total"] - float(fwd(x) - fwd(torch.zeros_like(x)))) <= res["convergence_delta"] + 1e-3
