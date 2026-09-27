"""v0.3 interventions: the edit set, capture-time hooks (TorchLens), NNsight edits, storage and HTTP.

References are forward hooks on the live model (``hooked`` from test_attr_m3)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from test_attr_m3 import finish, hooked
from test_pipeline import Tiny
from torch import nn

from neurotome import capture
from neurotome.api import create_app
from neurotome.graph import module_site
from neurotome.interventions import (
    diff_row,
    downstream,
    edit_fn,
    index_list,
    keep_ops,
    label,
    normalise,
    set_hash,
)
from neurotome.models import to_logits
from neurotome.runs import RunManager
from neurotome.store import RunStore

TINY = "py:test_pipeline:Tiny"
META_KEYS = {
    "hash",
    "run_id",
    "name",
    "created",
    "set",
    "labels",
    "method_label",
    "target",
    "target_label",
    "before",
    "after",
    "delta_logit",
    "delta_prob",
    "top_before",
    "top_after",
    "edited",
    "downstream",
    "saved",
    "truncated",
    "budget_bytes",
    "bytes",
    "nodes",
    "stats",
}


def op_(oid, shape, dtype="torch.float32", io=None, children=()):
    return {
        "id": oid,
        "shape": shape,
        "dtype": dtype,
        "is_input": io == "in",
        "is_output": io == "out",
        "children": list(children),
        "saved": True,
    }


IR = {
    "ops": [
        op_("input_1", [1, 3, 5, 6], io="in", children=["conv"]),
        op_("conv", [1, 4, 5, 6], children=["seq"]),
        op_("seq", [1, 7, 8], children=["vec"]),
        op_("vec", [1, 8], children=["ids"]),
        op_("ids", [1, 7], "torch.int64"),
        op_("odd", [5]),
        op_("output_1", [1, 8], io="out"),
    ]
}


def one(**e):
    return normalise({"edits": [e]}, IR)["edits"][0]


# ---- normalise, hash, labels (no model) ------------------------------------------------------------


def test_normalise_and_hash():
    a = normalise(
        {
            "edits": [
                {
                    "op": "conv",
                    "kind": "zero",
                    "channels": [3, 1, 1],
                    "positions": [[2, 1], [0, 0]],
                    "alpha": 5,
                    "value": 2,
                    "seed": 4,
                }
            ],
            "target": None,
        },
        IR,
    )
    assert a == {
        "target": None,
        "edits": [{"op": "conv", "kind": "zero", "channels": [1, 3], "positions": [[0, 0], [2, 1]]}],
    }
    b = normalise(
        {
            "target": None,
            "edits": [
                {"positions": [(0, 0), [2, 1], [0, 0]], "channels": [1, 3, 3], "kind": "zero", "op": "conv"}
            ],
        },
        IR,
    )
    assert set_hash(a) == set_hash(b) and re.fullmatch(r"[0-9a-f]{12}", set_hash(a))
    assert set_hash(a) != set_hash(normalise({"edits": [{"op": "conv", "kind": "zero"}]}, IR))
    assert one(op="seq", kind="mean") == {
        "op": "seq",
        "kind": "mean",
        "channels": None,
        "positions": None,
        "over": "positions",
    }
    assert one(op="seq", kind="scale", channels=[])["alpha"] == 1.0
    assert one(op="seq", kind="constant", positions="last") | {} == {
        "op": "seq",
        "kind": "constant",
        "channels": None,
        "positions": "last",
        "value": 0.0,
    }
    assert one(op="seq", kind="resample", source_run="random")["seed"] == 0
    assert "seed" not in one(op="seq", kind="resample", source_run="run_0002", seed=3)
    contrast = one(
        op="seq",
        kind="steer",
        direction={"kind": "contrast", "run_a": "run_0001", "run_b": "run_0002", "channel": 4},
    )
    assert contrast["direction"] == {
        "kind": "contrast",
        "run_a": "run_0001",
        "run_b": "run_0002",
        "pool": "mean",
    }
    ch = one(op="vec", kind="steer", alpha=4, direction={"kind": "channel", "channel": 3, "pool": "last"})
    assert ch["direction"] == {"kind": "channel", "channel": 3} and ch["alpha"] == 4.0
    assert normalise({"edits": [{"op": "conv", "kind": "zero"}], "target": 3}, IR)["target"] == 3


@pytest.mark.parametrize(
    ("edits", "msg"),
    [
        ([], "An intervention set needs at least one edit"),
        ([{"op": "conv", "kind": "zero"}] * 33, "At most 32 edits per set"),
        ([{"op": "nope", "kind": "zero"}], "Unknown op nope"),
        (
            [{"op": "input_1", "kind": "zero"}],
            "input_1 is the model's input or output; pick an op inside the model",
        ),
        ([{"op": "ids", "kind": "zero"}], "ids holds torch.int64 values; only float tensors can be edited"),
        ([{"op": "conv", "kind": "zero", "channels": [4]}], "Channel 4 out of range for conv (4 channels)"),
        ([{"op": "seq", "kind": "zero", "channels": [8]}], "Channel 8 out of range for seq (8 channels)"),
        (
            [{"op": "odd", "kind": "zero", "channels": [0]}],
            "odd has shape [5]: channel / position scopes need a (B, C, H, W), (B, T, D) or (B, D) tensor",
        ),
        (
            [{"op": "vec", "kind": "zero", "positions": [0]}],
            "vec has no positions (shape [1, 8]); scope it by channels",
        ),
        (
            [{"op": "conv", "kind": "zero", "positions": "last"}],
            '"last" applies to token positions (B, T, D)',
        ),
        (
            [{"op": "conv", "kind": "zero", "positions": [2]}],
            "Positions of conv are [y, x] pixels (H 5, W 6)",
        ),
        (
            [{"op": "conv", "kind": "zero", "positions": [[5, 0]]}],
            "Positions of conv are [y, x] pixels (H 5, W 6)",
        ),
        (
            [{"op": "seq", "kind": "zero", "positions": [[1, 2]]}],
            "Positions of seq are token indices below 7",
        ),
        ([{"op": "seq", "kind": "zero", "positions": [7]}], "Positions of seq are token indices below 7"),
        ([{"op": "seq", "kind": "resample"}], 'Resample needs source_run (a run id or "random")'),
        ([{"op": "seq", "kind": "steer"}], "Steering needs a direction"),
        (
            [{"op": "seq", "kind": "steer", "direction": {"kind": "contrast", "run_a": "run_0001"}}],
            "Steering needs a direction",
        ),
        (
            [{"op": "odd", "kind": "steer", "direction": {"kind": "channel", "channel": 0}}],
            "Steering needs a (B, C, H, W), (B, T, D) or (B, D) tensor; odd is [5]",
        ),
        (
            [{"op": "vec", "kind": "steer", "direction": {"kind": "channel", "channel": 8}}],
            "Direction channel 8 out of range for vec (8 channels)",
        ),
    ],
)
def test_normalise_messages(edits, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        normalise({"edits": edits}, IR)


def test_labels():
    assert index_list([0, 3, 7, 8, 9]) == "0,3,7–9" and index_list([1, 2]) == "1,2"
    zero = one(op="conv", kind="zero", channels=[3, 0, 2, 1], positions=[[4, 1], [0, 0]])
    assert label(zero, "block.conv2") == "Zero · block.conv2 · ch 0–3 · px (0,0) (4,1)"
    many = one(op="conv", kind="zero", positions=[[0, x] for x in range(6)])
    assert label(many, "c").endswith("px (0,0) (0,1) (0,2) (0,3) +2")
    assert label(one(op="seq", kind="mean"), "p") == "Mean over positions · p · all"
    assert label(one(op="seq", kind="mean", over="runs"), "p", n_runs=3) == "Mean over 3 runs · p · all"
    assert label(one(op="seq", kind="resample", source_run="run_0004"), "p") == "Replace ← run_0004 · p · all"
    rnd = one(op="seq", kind="resample", source_run="random")
    assert label(rnd, "p", "run_0004") == "Resample ← run_0004 (random · seed 0) · p · all"
    assert (
        label(one(op="seq", kind="constant", value=0.5, positions="last"), "p")
        == "Constant 0.5 · p · last token"
    )
    assert (
        label(one(op="seq", kind="scale", alpha=1.25, positions=[2, 5]), "p")
        == "Scale ×1.25 · p · tokens 2,5"
    )
    assert label(one(op="seq", kind="scale", alpha=2), "p") == "Scale ×2 · p · all"
    steer = one(
        op="seq",
        kind="steer",
        alpha=4,
        direction={"kind": "contrast", "run_a": "run_0005", "run_b": "run_0006", "pool": "last"},
    )
    assert label(steer, "p") == "Steer +4 × (run_0006 − run_0005, last) · p · all"
    neg = one(op="seq", kind="steer", alpha=-0.5, direction={"kind": "channel", "channel": 3})
    assert label(neg, "p") == "Steer −0.5 × e_3 · p · all"  # the client's minus sign (U+2212)


def test_scope_and_kinds():
    x = torch.randn(2, 4, 3, 5)
    z = edit_fn(one(op="conv", kind="zero", channels=[1]), None)(x)
    assert (z[:, 1] == 0).all() and torch.equal(z[:, [0, 2, 3]], x[:, [0, 2, 3]])
    px = edit_fn(one(op="conv", kind="zero", positions=[[0, 1], [2, 4]]), None)(x)
    hit = torch.zeros(3, 5, dtype=torch.bool)
    hit[0, 1] = hit[2, 4] = True
    assert (px[..., hit] == 0).all() and torch.equal(px[..., ~hit], x[..., ~hit])
    both = edit_fn(one(op="conv", kind="constant", value=0.5, channels=[2], positions=[[1, 1]]), None)(x)
    assert (both[:, 2, 1, 1] == 0.5).all() and (both != x).sum() == 2
    seq = torch.randn(1, 6, 8)
    tok = edit_fn(one(op="seq", kind="zero", positions=[2, 5]), None)(seq)
    assert (tok[0, [2, 5]] == 0).all() and torch.equal(tok[0, [0, 1, 3, 4]], seq[0, [0, 1, 3, 4]])
    last = edit_fn(one(op="seq", kind="zero", positions="last"), None)
    assert (last(seq)[0, 5] == 0).all() and torch.equal(last(seq)[0, :5], seq[0, :5])
    longer = torch.randn(1, 7, 8)
    assert (last(longer)[0, 6] == 0).all() and torch.equal(
        last(longer)[0, :6], longer[0, :6]
    )  # rebuilt per call
    mean = edit_fn(one(op="conv", kind="mean"), None)
    assert torch.allclose(mean(x), x.mean((2, 3), keepdim=True).expand_as(x))
    assert torch.allclose(
        edit_fn(one(op="seq", kind="mean"), None)(seq), seq.mean(1, keepdim=True).expand_as(seq)
    )
    v = torch.randn(2, 8)
    assert torch.allclose(edit_fn(one(op="vec", kind="mean"), None)(v), v.mean(1, keepdim=True).expand_as(v))
    assert torch.equal(edit_fn(one(op="conv", kind="scale", alpha=1), None)(x), x)
    steer0 = one(op="seq", kind="steer", alpha=0, direction={"kind": "channel", "channel": 3})
    assert torch.equal(edit_fn(steer0, None)(seq), seq)
    steer = edit_fn(one(op="seq", kind="steer", alpha=2, direction={"kind": "channel", "channel": 3}), None)(
        seq
    )
    assert torch.allclose(steer[..., 3], seq[..., 3] + 2) and torch.equal(steer[..., :3], seq[..., :3])
    vec = np.arange(8, dtype=np.float32)
    contrast = one(
        op="seq", kind="steer", alpha=0.5, direction={"kind": "contrast", "run_a": "a", "run_b": "b"}
    )
    assert torch.allclose(edit_fn(contrast, vec)(seq), seq + 0.5 * torch.from_numpy(vec))
    src = np.ones((1, 6, 8), np.float32)
    rs = edit_fn(one(op="seq", kind="resample", source_run="run_0002", positions=[0]), src)(seq.double())
    assert rs.dtype == torch.float64 and (rs[0, 0] == 1).all() and torch.equal(rs[0, 1:], seq[0, 1:].double())


def test_diff_row_and_graph_sets():
    a = np.array([[1.0, 2.0, np.nan], [3.0, 0.0, 1.0]], np.float32)
    same = diff_row(a, a)
    assert same["l2"] == 0 and same["rel"] == 0 and same["cos"] == pytest.approx(1) and same["max_abs"] == 0
    z = diff_row(a, np.zeros_like(a))
    assert z["rel"] == pytest.approx(1) and z["cos"] is None and z["max_abs"] == 3
    assert diff_row(np.zeros(3), np.ones(3)) == {
        "l2": pytest.approx(3**0.5),
        "rel": 0.0,
        "cos": None,
        "max_abs": 1,
        "nonfinite": 0,
    }
    inf = diff_row(np.ones(4), np.array([np.inf, np.nan, 1, 1]))  # all finite change is in the NaN / inf
    assert inf["nonfinite"] == 2 and inf["l2"] == 0
    assert diff_row(a, a.copy())["nonfinite"] == 0  # NaN before and after: unchanged
    assert downstream(IR, ["seq"]) == ["seq", "vec", "ids"]
    assert keep_ops(IR, ["conv", "seq", "vec", "ids"], None) == (["conv", "seq", "vec"], False)
    assert keep_ops(IR, ["conv", "seq", "vec"], 4 * (120 + 56)) == (["conv", "seq"], True)  # never crosses
    assert keep_ops(IR, ["conv", "seq", "vec"], 100) == ([], True)


# ---- TorchLens: Tiny ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("iv") / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")["id"]
    live = mgr.live(run)
    return mgr, run, mgr.store.ir(run), live.loaded.model, live.inputs


def iv(mgr, run, edits, target=None):
    return mgr.intervene(run, normalise({"edits": edits, "target": target}, mgr.store.ir(run)))


def hooks(model, x, fns):
    """model(x) with forward hooks {module: {call: fn}}."""
    counts, handles = {}, []
    for mod, calls in fns.items():

        def h(m, _i, out, calls=calls):
            counts[m] = counts.get(m, 0) + 1
            return calls[counts[m]](out) if counts[m] in calls else out

        handles.append(mod.register_forward_hook(h))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        for h in handles:
            h.remove()


def test_tiny_zero_stem(tiny):
    mgr, run, ir, model, x = tiny
    res = iv(mgr, run, [{"op": "conv2d_1_1", "kind": "zero"}])
    ref = hooked(model, model.stem[0], x, torch.zeros_like)
    t = res["target"]
    assert set(res) == META_KEYS and res["run_id"] == run and res["edited"] == ["conv2d_1_1"]
    assert res["labels"] == ["Zero · stem.0 · all"] and res["name"] == "Zero · stem.0 · all"
    assert res["method_label"] == "Intervention · TorchLens capture-time hooks (every op)"
    assert res["after"]["logit"] == pytest.approx(float(ref[0, t]), abs=1e-4)
    assert res["delta_logit"] == pytest.approx(res["after"]["logit"] - res["before"]["logit"])
    assert len(res["top_before"]) == len(res["top_after"]) == 5 and not res["truncated"]
    assert "input_1" not in res["downstream"] and res["downstream"][0] == "conv2d_1_1"
    assert all(res["nodes"][op]["rel"] > 0 for op in res["downstream"])
    assert set(res["nodes"]) == set(res["stats"]) == set(res["downstream"])
    assert set(res["saved"]) <= set(res["downstream"]) and res["saved"]
    folder = mgr.store.iv_dir(run, res["hash"])
    assert (folder / "meta.json").is_file() and res["bytes"] == sum(
        np.load(folder / "tensors" / f"{op}.npy").nbytes for op in res["saved"]
    )
    assert not np.asarray(mgr.store.load_tensor(run, "conv2d_1_1", iv=res["hash"])).any()
    out = capture.intervene(model, x, [("conv2d_1_1", torch.zeros_like)], downstream(ir, ["conv2d_1_1"]))
    assert torch.allclose(out["logits_after"], ref, atol=1e-5) and out["after"] == {}


def test_tiny_noops(tiny):
    mgr, run, ir, model, x = tiny
    res = iv(mgr, run, [{"op": "conv2d_3_5", "kind": "scale", "alpha": 1}])
    assert all(n["l2"] == 0 for n in res["nodes"].values()) and res["after"] == res["before"]
    assert "conv2d_1_1" not in res["downstream"] and "add_1_6" in res["downstream"]
    same = iv(mgr, run, [{"op": "relu_2_4", "kind": "resample", "source_run": run}])
    assert same["labels"] == [f"Replace ← {run} · block.relu · all"]
    assert all(n["l2"] < 1e-6 for n in same["nodes"].values())


def test_tiny_overflow(tiny):
    """An edit that overflows to inf / NaN: null logits and probs (strict JSON), nodes flag the NaN / inf."""
    mgr, run, *_ = tiny
    res = iv(mgr, run, [{"op": "conv2d_3_5", "kind": "scale", "alpha": 1e38}] * 2)
    assert res["after"] == {"logit": None, "prob": None} and res["delta_logit"] is None
    assert res["before"]["logit"] is not None and all(t["prob"] is None for t in res["top_after"])
    assert res["nodes"]["conv2d_3_5"]["nonfinite"] > 0 and res["nodes"]["output_1"]["nonfinite"] > 0
    text = (mgr.store.iv_dir(run, res["hash"]) / "meta.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_tiny_two_edits(tiny):
    mgr, run, ir, model, x = tiny
    edits = [
        {"op": "conv2d_2_3", "kind": "zero", "channels": [0, 1, 2, 3]},
        {"op": "relu_3_7", "kind": "scale", "alpha": 3},
    ]
    res = iv(mgr, run, edits)
    assert res["labels"] == ["Zero · block.conv1 · ch 0–3", "Scale ×3 · block.relu@2 · all"]
    assert res["name"] == "Zero · block.conv1 · ch 0–3 (+1)" and res["edited"] == ["conv2d_2_3", "relu_3_7"]

    def zero03(t):
        t = t.clone()
        t[:, :4] = 0
        return t

    seen = {}
    ref = hooks(
        model,
        x,
        {model.block.conv1: {1: zero03}, model.block.relu: {2: lambda t: seen.setdefault("r", t * 3)}},
    )
    assert res["after"]["logit"] == pytest.approx(float(ref[0, res["target"]]), abs=1e-4)
    fns = [(e["op"], edit_fn(e, None)) for e in normalise({"edits": edits}, ir)["edits"]]
    out = capture.intervene(model, x, fns, downstream(ir, ["conv2d_2_3"]), keep=["relu_3_7"])
    assert torch.allclose(out["logits_after"], ref, atol=1e-5)
    assert np.allclose(out["after"]["relu_3_7"], seen["r"].numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="Unknown op nope"):
        capture.intervene(model, x, [("nope", torch.zeros_like)], [])
    with pytest.raises(
        ValueError, match=re.escape("Edit of conv2d_3_5 changed its shape [1, 8, 16, 16] -> ")
    ):
        capture.intervene(model, x, [("conv2d_3_5", lambda t: t[:, :2])], [])
    with pytest.raises(ValueError, match=re.escape("Target 9 is outside the output (0–4)")):
        iv(mgr, run, [{"op": "conv2d_3_5", "kind": "zero"}], target=9)


class Fork(nn.Module):
    """lin's output feeds a second consumer before the dropout (so TorchLens credits it to lin, not to the
    dropout, which returns the same object): editing lin is not editing the dropout."""

    def __init__(self):
        super().__init__()
        self.lin, self.drop = nn.Linear(4, 8), nn.Dropout(0.1)
        self.head, self.side = nn.Linear(8, 3), nn.Linear(8, 3)

    def forward(self, x):
        h = self.lin(x)
        s = self.side(h)
        return self.head(self.drop(h)) + s


def test_dropout_edits_its_input(tmp_path):
    """Eval dropout returns its input object, which TorchLens keeps: the edit moves to that input when the
    dropout is its only consumer (the result is then exactly the edit), and is refused otherwise."""
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.1), nn.Linear(8, 3))
    run = mgr.create_from_objects(model, torch.randn(1, 4))["id"]
    ir = mgr.store.ir(run)
    drop = next(o for o in ir["ops"] if o["type"] == "dropout")
    res = iv(mgr, run, [{"op": drop["id"], "kind": "scale", "alpha": 0.5}])
    ref = iv(mgr, run, [{"op": drop["parents"][0], "kind": "scale", "alpha": 0.5}])
    assert res["delta_logit"] == pytest.approx(ref["delta_logit"], abs=1e-6) and res["delta_logit"] != 0
    assert res["nodes"][drop["id"]]["rel"] == pytest.approx(0.5, abs=1e-6)
    fork = mgr.create_from_objects(Fork(), torch.randn(1, 4))["id"]
    ir = mgr.store.ir(fork)
    drop = next(o for o in ir["ops"] if o["type"] == "dropout")
    with pytest.raises(ValueError) as e:
        iv(mgr, fork, [{"op": drop["id"], "kind": "zero"}])
    assert str(e.value) == (
        f"TorchLens cannot edit {drop['id']} (dropout): it returns its input unchanged in "
        f"eval mode. Edit its parent {drop['parents'][0]} instead."
    )


def test_resnet18_inplace(tmp_path):
    """Zeroing layer3.0's downsample output: `out += identity` and ReLU(inplace) after it stay exact."""
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    run = mgr.create_from_spec("torchvision:resnet18", None, {"kind": "random", "shape": [1, 3, 64, 64]})[
        "id"
    ]
    ir, live = mgr.store.ir(run), mgr.live(run)
    model, x = live.loaded.model, live.inputs
    op = next(o["id"] for o in ir["ops"] if o["modules"][-1:] == ["layer3.0.downsample.1:1"])
    seen = {}
    h = model.layer3[0].register_forward_hook(
        lambda _m, _i, out: seen.__setitem__("block", out.detach().clone())
    )
    try:
        ref = hooked(model, model.layer3[0].downsample, x, torch.zeros_like)
    finally:
        h.remove()
    out = capture.intervene(model, x, [(op, torch.zeros_like)], downstream(ir, [op]), keep=["relu_11_43"])
    assert torch.allclose(out["logits_after"], ref, atol=1e-4)
    assert np.allclose(out["after"]["relu_11_43"], seen["block"].numpy(), atol=1e-5)
    assert out["nodes"]["iadd_5_42"]["rel"] > 0  # replay lost it; capture-time hooks record it
    bn2 = next(o["id"] for o in ir["ops"] if o["modules"][-1:] == ["layer3.0.bn2:1"])
    assert bn2 not in out["nodes"]  # beside the edited op, not after it
    # bn2 feeds `out += identity`: an expanded view as the replacement would raise
    mean = edit_fn(one(op="conv", kind="mean"), None)
    ref = hooked(model, model.layer3[0].bn2, x, mean)
    out = capture.intervene(model, x, [(bn2, mean)], downstream(ir, [bn2]))
    assert torch.allclose(out["logits_after"], ref, atol=1e-4)


# ---- TorchLens: tiny GPT-2 (replay got the residual adds wrong) ---------------------------------------


@pytest.fixture(scope="module")
def gpt2(tmp_path_factory):
    from test_attention_v02 import tiny_gpt2

    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("ivlm") / "runs"))
    model = tiny_gpt2()
    run = mgr.create_from_objects(model, torch.randint(0, 64, (1, 7)))["id"]
    ir = mgr.store.ir(run)
    block0 = next(o["id"] for o in ir["ops"] if module_site(ir, o["id"]) == ("transformer.h.0", 1))
    return mgr, run, ir, model, mgr.live(run).inputs, block0


@pytest.mark.parametrize(
    ("edit", "ref"),
    [
        ({"kind": "scale", "alpha": 2}, lambda t: t * 2),
        ({"kind": "zero", "positions": [2]}, lambda t: torch.cat([t[:, :2], 0 * t[:, 2:3], t[:, 3:]], 1)),
        ({"kind": "mean"}, lambda t: t.mean(1, keepdim=True).expand_as(t)),
    ],
)
def test_gpt2_block_output(gpt2, edit, ref):
    from neurotome.interventions import _apply

    mgr, run, ir, model, x, block0 = gpt2
    e = normalise({"edits": [{"op": block0, **edit}]}, ir)["edits"][0]
    out = capture.intervene(model, x, [(block0, edit_fn(e, None))], downstream(ir, [block0]))
    want = to_logits(hooked(model, model.transformer.h[0], x, lambda o: _apply(o, ref)))
    assert torch.allclose(to_logits(out["logits_after"]), want, atol=1e-5)
    assert out["nodes"]["output_1"]["rel"] > 0


def test_gpt2_resample_self_noop(gpt2):
    mgr, run, ir, _model, _x, block0 = gpt2
    res = iv(mgr, run, [{"op": block0, "kind": "resample", "source_run": run}])
    assert all(n["l2"] < 1e-6 for n in res["nodes"].values()) and res["after"]["logit"] == pytest.approx(
        res["before"]["logit"], abs=1e-5
    )
    assert iv(mgr, run, [{"op": block0, "kind": "resample", "source_run": run}]) == res  # cached


# ---- NNsight -------------------------------------------------------------------------------------


def test_nnsight_edits(tmp_path):
    pytest.importorskip("nnsight")
    from test_attention_v02 import tiny_gpt2

    from neurotome import capture_nnsight as cn

    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), backend="nnsight")["id"]
    ir, live = mgr.store.ir(run), mgr.live(run)
    model, x = live.loaded.model, live.inputs
    down = downstream(ir, ["block.conv2@1"])
    fns = [("block.relu@2", lambda t: t * 3), ("block.conv2@1", torch.zeros_like)]  # any order
    out = cn.intervene(model, x, fns, down, keep=["block.relu@2"])
    ref = hooks(model, x, {model.block.conv2: {1: torch.zeros_like}, model.block.relu: {2: lambda t: t * 3}})
    assert torch.allclose(to_logits(out["logits_after"]), ref, atol=1e-5)
    assert set(out["nodes"]) == set(down) and out["nodes"]["block.conv2@1"]["rel"] == pytest.approx(1)
    seen = {}
    hooks(
        model,
        x,
        {
            model.block.conv2: {1: torch.zeros_like},
            model.block.relu: {2: lambda t: seen.setdefault("r", t * 3)},
        },
    )
    assert np.allclose(out["after"]["block.relu@2"], seen["r"].numpy(), atol=1e-6)
    noop = cn.intervene(model, x, [("block.relu@1", lambda t: t * 1)], downstream(ir, ["block.relu@1"]))
    assert all(n["l2"] == 0 for n in noop["nodes"].values())
    with pytest.raises(ValueError, match="Unknown op block@1"):
        cn.intervene(model, x, [("block@1", torch.zeros_like)], [])
    with pytest.raises(ValueError, match="Unknown op block.relu@3"):
        cn.intervene(model, x, [("block.relu@3", torch.zeros_like)], [])
    res = iv(mgr, run, [{"op": "block.relu@2", "kind": "scale", "alpha": 3}])
    assert res["labels"] == ["Scale ×3 · block.relu@2 · all"]
    assert res["after"]["logit"] == pytest.approx(
        float(hooks(model, x, {model.block.relu: {2: lambda t: t * 3}})[0, res["target"]]), abs=1e-4
    )

    same = iv(mgr, run, [{"op": "block.relu@2", "kind": "resample", "source_run": run}])
    assert all(n["l2"] < 1e-6 for n in same["nodes"].values())
    px = [[0, 0], [3, 5]]
    res = iv(mgr, run, [{"op": "block.conv2@1", "kind": "zero", "channels": [0, 1], "positions": px}])

    def zero_px(t):
        t = t.clone()
        t[:, :2, [0, 3], [0, 5]] = 0
        return t

    ref = hooks(model, x, {model.block.conv2: {1: zero_px}})
    assert res["labels"] == ["Zero · block.conv2 · ch 0,1 · px (0,0) (3,5)"]
    assert res["after"]["logit"] == pytest.approx(float(ref[0, res["target"]]), abs=1e-4)
    assert 0 < res["nodes"]["block.conv2@1"]["rel"] < 1
    for edit, fn in (
        ({"kind": "mean"}, lambda t: t.mean((2, 3), keepdim=True).expand_as(t)),
        ({"kind": "constant", "value": 0.5}, lambda t: torch.full_like(t, 0.5)),
    ):
        res = iv(mgr, run, [{"op": "block.conv2@1", **edit}])
        ref = hooks(model, x, {model.block.conv2: {1: fn}})
        assert res["after"]["logit"] == pytest.approx(float(ref[0, res["target"]]), abs=1e-4)

    lm = tiny_gpt2().eval()
    ids = torch.randint(0, 64, (1, 7))
    site = "transformer.h.0.mlp.c_proj@1"
    e = {
        "op": site,
        "kind": "steer",
        "alpha": 5.0,
        "channels": None,
        "positions": None,
        "direction": {"kind": "channel", "channel": 3},
    }
    out = cn.intervene(lm, ids, [(site, edit_fn(e, None))], [site, "output_1"])
    want = hooked(lm, lm.transformer.h[0].mlp.c_proj, ids, lambda t: t + 5 * torch.eye(32)[3])
    assert torch.allclose(to_logits(out["logits_after"]), to_logits(want), atol=1e-5)


# ---- RunManager: jobs, storage, budget, source runs -----------------------------------------------


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    """Three runs of py:test_pipeline:Tiny (same spec, different inputs) and a manager."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"), max_live_models=4)
    ids = [
        mgr.create_from_spec(TINY, None, {"kind": "random", "shape": [1, 3, 16, 16], "seed": s})["id"]
        for s in (1, 2, 3)
    ]
    return mgr, ids


def test_jobs_cache_and_budget(runs):
    mgr, (a, *_rest) = runs
    body = {"edits": [{"op": "conv2d_2_3", "kind": "zero", "channels": [1, 0]}], "target": None}
    job = finish(mgr, mgr.job(a, {"kind": "intervene", "interventions": body}))
    assert job["status"] == "done", job["error"]
    res = job["result"]
    permuted = {"target": None, "edits": [{"channels": [0, 1], "kind": "zero", "op": "conv2d_2_3"}]}
    again = mgr.job(a, {"kind": "intervene", "interventions": permuted})
    assert again.status == "done" and again.result == res
    assert [r["hash"] for r in mgr.interventions(a)] == [res["hash"]]
    ir = mgr.store.ir(a)
    mgr.max_capture_bytes = budget = 3 * 8 * 16 * 16 * 4 + 10  # three conv-sized tensors
    small = iv(mgr, a, [{"op": "conv2d_1_1", "kind": "scale", "alpha": 2}])
    eligible = [op for op in small["downstream"] if next(o for o in ir["ops"] if o["id"] == op)["saved"]]
    assert small["truncated"] and small["budget_bytes"] == budget and small["bytes"] <= budget
    assert small["saved"] == eligible[: len(small["saved"])] and 0 < len(small["saved"]) < len(eligible)
    assert set(small["nodes"]) == set(small["downstream"])  # every Δ, saved or not
    with pytest.raises(KeyError, match="capture budget"):
        mgr.iv_tensor(a, small["hash"], eligible[-1])


def test_source_runs(runs, tmp_path):
    mgr, (a, b, c) = runs
    op = "relu_2_4"

    def post(edit):
        return mgr.job(a, {"kind": "intervene", "interventions": {"edits": [{"op": op, **edit}]}})

    rnd = {"kind": "resample", "source_run": "random", "seed": 0}
    first = finish(mgr, post(rnd))["result"]
    chosen = first["labels"][0].split("← ")[1].split(" ")[0]
    assert (
        chosen in (b, c) and first["labels"][0] == f"Resample ← {chosen} (random · seed 0) · block.relu · all"
    )
    assert mgr.resolve(a, first["set"])[0][1] == chosen  # reproducible for a seed
    saved = np.asarray(mgr.store.load_tensor(a, op, iv=first["hash"]))
    assert np.allclose(saved, np.asarray(mgr.store.load_tensor(chosen, op)))
    mean = finish(mgr, post({"kind": "mean", "over": "runs"}))["result"]
    assert mean["labels"][0] == "Mean over 3 runs · block.relu · all"
    want = np.mean([np.asarray(mgr.store.load_tensor(r, op)) for r in (a, b, c)], axis=0)
    assert np.allclose(mgr.store.load_tensor(a, op, iv=mean["hash"]), want, atol=1e-6)

    other = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))["id"]  # in-process: another model
    small = mgr.create_from_spec(TINY, None, {"kind": "random", "shape": [1, 3, 8, 8]})["id"]
    meta = mgr.store.meta(c)
    meta["model"]["weights"] = "DEFAULT"
    mgr.store.write_json(c, "meta.json", meta)
    mgr.store.tensor_path(b, op).unlink()
    cases = [
        ({"kind": "resample", "source_run": "run_0099"}, "Source run run_0099 not found"),
        (
            {"kind": "resample", "source_run": other},
            f"{other} is a run of another model (None · None); use a run",
        ),
        ({"kind": "resample", "source_run": c}, f"{c} is a run of another model ({TINY} · DEFAULT)"),
        ({"kind": "resample", "source_run": b}, f"{b} has no saved tensor for {op}"),
        (
            {"kind": "resample", "source_run": small},
            f"{small}'s {op} has shape [1, 8, 8, 8], this run's [1, 8, 16, 16]",
        ),
        (
            {"kind": "resample", "source_run": "random"},
            f"No other run of this model has a saved tensor for {op}",
        ),
        (
            {"kind": "mean", "over": "runs"},
            f"Mean over runs needs another saved run of this model with a tensor for {op}",
        ),
        (
            {"kind": "steer", "direction": {"kind": "contrast", "run_a": a, "run_b": other}},
            "is a run of another model",
        ),
    ]
    for edit, msg in cases:
        with pytest.raises(ValueError, match=re.escape(msg)):
            post({**edit, "channels": [0]})  # not a set that already ran (a cached set is born done)
    with pytest.raises(ValueError, match=re.escape("intervene jobs need interventions")):
        mgr.job(a, {"kind": "intervene"})


# ---- HTTP -----------------------------------------------------------------------------------------


def test_http_routes(runs):
    from test_jobs_m3 import poll

    mgr, (a, *_rest) = runs
    c = TestClient(create_app(mgr))
    url = f"/api/runs/{a}"
    body = {"kind": "intervene", "interventions": {"edits": [{"op": "conv2d_3_5", "kind": "zero"}]}}
    res = poll(c, c.post(f"{url}/jobs", json=body).json())["result"]
    h, op = res["hash"], "conv2d_3_5"
    assert c.post(f"{url}/jobs", json=body).json()["status"] == "done"
    rows = c.get(f"{url}/interventions").json()
    assert [r["hash"] for r in rows] == [h] and set(rows[0]) == {
        "hash",
        "name",
        "created",
        "labels",
        "target",
        "target_label",
        "delta_logit",
        "delta_prob",
        "truncated",
    }
    assert c.get(f"{url}/interventions/{h}").json() == res
    r = c.patch(f"{url}/interventions/{h}", json={"name": "  ablate conv "})
    assert r.status_code == 200 and r.json()["name"] == "ablate conv"
    assert c.get(f"{url}/interventions").json()[0]["name"] == "ablate conv"
    r = c.patch(f"{url}/interventions/{h}", json={"name": "   "})
    assert r.status_code == 400 and r.json()["detail"] == "Name cannot be empty"
    assert c.patch(f"{url}/interventions/{h}", json={"name": "x" * 81}).status_code == 422

    t = c.get(f"{url}/ops/{op}/tensor", params={"iv": h}).json()
    assert t["stats"]["max"] == 0 and t["stats"]["min"] == 0
    clean = c.get(f"{url}/ops/conv2d_1_1/tensor").json()
    assert c.get(f"{url}/ops/conv2d_1_1/tensor", params={"iv": h}).json() == clean  # not downstream
    sl = c.get(f"{url}/ops/{op}/slice", params={"s": "[0, 0, 0, :4]", "iv": h}).json()
    assert sl["values"] == [0, 0, 0, 0]
    assert c.get(f"{url}/ops/{op}/slice.npy", params={"iv": h}).status_code == 200
    d = c.get(f"{url}/diff", params={"a": "add_1_6", "b": "add_1_6", "b_iv": h}).json()
    assert d["diff"]["rel_l2"] == pytest.approx(res["nodes"]["add_1_6"]["rel"], rel=1e-4)
    g = c.get(f"{url}/graph", params={"depth": 3, "iv": h}).json()
    node = next(n for n in g["nodes"] if n["output_op"] == op)
    assert node["stats"]["max"] == 0 and node["stats"] == res["stats"][op]
    for path in (f"{url}/ops/{op}/tensor", f"{url}/graph", f"{url}/interventions/abc"):
        r = c.get(path, params={"iv": "0123456789ab"} if "interventions" not in path else None)
        assert r.status_code == 404 and "not found in" in r.json()["detail"]
    assert c.get(f"{url}/ops/{op}/tensor", params={"iv": "../../x"}).status_code == 404
    mgr.store.tensor_path(a, "relu_3_7", iv=h).unlink()
    mgr.store.write_json(a, f"interventions/{h}/meta.json", res | {"saved": [op], "truncated": True})
    r = c.get(f"{url}/ops/relu_3_7/tensor", params={"iv": h})
    assert r.status_code == 404 and r.json()["detail"].startswith(
        "No intervened tensor saved for relu_3_7: the capture budget"
    )

    jobs = f"{url}/jobs"
    assert c.post(jobs, json={"kind": "patch", "op": op}).status_code == 422
    assert c.post(jobs, json={**body, "baseline": "zeros"}).status_code == 422
    assert (
        c.post(
            jobs, json={"kind": "intervene", "interventions": {"edits": [{"op": op, "kind": "x"}]}}
        ).status_code
        == 422
    )
    r = c.post(
        jobs,
        json={"kind": "intervene", "interventions": {"edits": [{"op": op, "kind": "zero", "channels": [8]}]}},
    )
    assert r.status_code == 400 and r.json()["detail"] == "Channel 8 out of range for conv2d_3_5 (8 channels)"
    assert c.post(jobs, json={"kind": "intervene"}).json()["detail"] == "intervene jobs need interventions"
    assert c.delete(f"{url}/interventions/{h}").json() == {"deleted": True}
    r = c.get(f"{url}/interventions/{h}")
    assert r.status_code == 404 and r.json()["detail"] == f"Intervention {h} not found in {a}"
    assert c.delete(f"{url}/interventions/{h}").status_code == 404
    assert (
        c.get(f"{url}/interventions").json() == []
        and c.get("/api/runs/run_9999/interventions").status_code == 404
    )


def test_mha_fast_path(tmp_path):
    """Under a no_grad caller nn.MultiheadAttention must not take its fused path (op ids would differ)."""
    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path / "runs"))
    layer = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
    run = mgr.create_from_objects(layer, torch.randn(1, 5, 8))["id"]
    ir, live = mgr.store.ir(run), mgr.live(run)
    op = next(o["id"] for o in ir["ops"] if o["id"].startswith("linear"))
    with torch.no_grad():
        out = capture.intervene(
            live.loaded.model, live.inputs, [(op, torch.zeros_like)], downstream(ir, [op])
        )
    assert out["nodes"][op]["rel"] == 1
