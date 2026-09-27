"""0.3.0 review: the channels of a (B, T, D) tensor are its features everywhere, as in interventions."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from neurotome.graph import module_site
from neurotome.runs import RunManager, okey
from neurotome.store import RunStore


class Seq(nn.Module):
    """(1, T=5, 4) floats -> (1, 5, D=6) -> logits over 3 classes; T != D so a token index would show."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(4, 6)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        return self.head(torch.tanh(self.proj(x)).mean(1))


@pytest.fixture()
def seq(tmp_path):
    mgr = RunManager(RunStore(tmp_path / "runs"))
    x = torch.randn(1, 5, 4, generator=torch.Generator().manual_seed(1))
    run = mgr.create_from_objects(Seq(), x)["id"]
    ir = mgr.store.ir(run)
    op = next(o for o in ir["ops"] if module_site(ir, o["id"]) == ("proj", 1))
    assert op["shape"] == [1, 5, 6]
    return mgr, run, op


def test_unit_objective_and_channel_scores_use_features(seq):
    mgr, run, op = seq
    obj = {"op": op["id"], "channel": 5}  # a token index would be out of range (T = 5)
    res = mgr.attribution(run, "gxa", None, 8, obj)
    act = np.asarray(mgr.store.load_tensor(run, op["id"]), np.float64)
    assert res["value"] == pytest.approx(act[0, :, 5].sum(), rel=1e-4, abs=1e-5)
    assert res["objective_label"] == "proj · ch 5 · Σ positions" and res["seq_channels"] == "features"
    top = mgr.attribution(run, "gxa", None, 8)
    channels = mgr.store.read_json(run, f"attribution/channels_{okey(None, None)}.json")
    assert len(channels[op["id"]]) == 6
    assert all(u["channel"] < 6 for u in top["top_units"])


def test_layer_and_neuron_methods_use_features(seq):
    mgr, run, op = seq
    res = mgr.layer(run, "layer_ig", op["id"], None, 16)
    assert len(res["channels"]) == 6
    assert abs(res["total"] - sum(res["channels"])) < 1e-4
    assert mgr.neuron(run, op["id"], 5, None, 8)["channel"] == 5
    with pytest.raises(ValueError, match="out of range"):
        mgr.neuron(run, op["id"], 6, None, 8)


def test_cached_results_without_the_stamp_are_recomputed(seq):
    mgr, run, _ = seq
    res = mgr.attribution(run, "gxa", None, 8)
    name = "attribution/gxa_top_8.json"
    mgr.store.write_json(run, name, {k: v for k, v in res.items() if k != "seq_channels"})
    assert mgr._cached(run, name) is None
    assert mgr.attribution(run, "gxa", None, 8)["seq_channels"] == "features"
