"""Input attribution of a sequence input: a channel x time map with block shares and time totals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neurotome.attribution import input_map
from neurotome.runs import INPUT_MAP, RunManager
from neurotome.store import RunStore

REPO = Path(__file__).resolve().parents[2]
SPEC = "py:examples.signal_model:build"


@pytest.fixture(scope="module")
def signal(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(REPO))
        mgr = RunManager(RunStore(tmp_path_factory.mktemp("ts") / "runs"))
        run = mgr.create_from_spec(SPEC, None, {"kind": "ref", "ref": "rec02:1"})
        yield mgr, run["id"]


def test_series_map_of_a_signal_run(signal):
    mgr, rid = signal
    res = mgr.attribution(rid, "gxa", None, 8)
    inp = res["input"]
    assert inp["kind"] == "series" and res["input_map"] == INPUT_MAP
    assert np.asarray(inp["map"]).shape == (10, 74)  # 10 feature channels x 74 steps, nothing pooled
    assert inp["channels"] == 10 and inp["steps"] == 74 and inp["stride"] == 1 and inp["row_stride"] == 1
    assert max(abs(v) for r in inp["map"] for v in r) == 1.0
    assert inp["time"] == {"t0": 608.0, "step_s": 8.0, "label": "s from recording start"}
    blocks = inp["blocks"]
    assert [b["name"] for b in blocks] == ["hr · band power", "pressure · band power"]
    assert blocks[1]["start"] == 5 and blocks[1]["labels"][0] == "0.0625–0.125 Hz"
    assert sum(b["share"] for b in blocks) == pytest.approx(1.0)
    assert sum(b["sum"] for b in blocks) == pytest.approx(inp["total"], rel=1e-4, abs=1e-6)
    assert len(inp["time_totals"]) == 74
    assert sum(inp["time_totals"]) == pytest.approx(inp["total"], rel=1e-4, abs=1e-6)
    # a neuron map goes through the same input_map
    op = next(o["id"] for o in mgr.store.ir(rid)["ops"] if o.get("module") == "conv.0")
    neuron = mgr.neuron(rid, op, 3, None, 4)["input"]
    assert neuron["kind"] == "series" and np.asarray(neuron["map"]).shape == (10, 74) and neuron["time"]
    # a stale (pre-series) cache entry is recomputed
    name = "attribution/gxa_top_8.json"
    old = mgr.store.read_json(rid, name)
    mgr.store.write_json(rid, name, {**old, "input_map": None})
    assert mgr._cached(rid, name) is None
    agree = mgr.agreement(rid, None, 4)
    assert agree["input"]["basis"] == "|attribution| · channel × time input map"


def test_pooling_raw_time_and_channel_axis():
    # a raw (1, C, T) signal: time on axis 2 (the longer one), the clock from the raw sampling rate
    attr = torch.randn(1, 2, 5280)
    raw = {"kind": "signal", "fs": 4.0, "t0": -1320.0, "time_label": "s to delivery",
           "channels": [{"name": "fhr", "unit": "bpm", "length": 5280},
                        {"name": "up", "unit": None, "length": 5280}]}
    m = input_map(attr, 0.1, {"raw": raw})
    assert np.asarray(m["map"]).shape == (2, 480) and m["stride"] == 11 and m["convergence_delta"] == 0.1
    assert m["time"] == {"t0": -1320.0, "step_s": 0.25, "label": "s to delivery"}
    assert [b["name"] for b in m["blocks"]] == ["fhr", "up"] and m["blocks"][0]["unit"] == "bpm"
    grid = attr[0].double().numpy()
    assert m["time_totals"][0] == pytest.approx(grid[:, :11].sum(), rel=1e-4)
    # (1, T, C) with many channels: channels pooled only above 160 rows, time only above 512 steps; no meta ->
    # one block, no clock
    m = input_map(torch.randn(1, 500, 400), None)
    assert np.asarray(m["map"]).shape == (134, 500) and m["row_stride"] == 3 and m["time"] is None
    assert [b["name"] for b in m["blocks"]] == ["input"] and m["blocks"][0]["share"] == pytest.approx(1.0)
    m = input_map(torch.randn(1, 300, 126), None)  # the teb model's input: a cell per step and channel
    assert np.asarray(m["map"]).shape == (126, 300) and m["row_stride"] == 1 and m["stride"] == 1


def test_image_and_vector_maps_unchanged():
    m = input_map(torch.randn(1, 3, 64, 64), None)
    assert set(m) == {"map", "total", "convergence_delta"} and np.asarray(m["map"]).shape == (24, 24)
    m = input_map(torch.randn(1, 30), None)
    assert "kind" not in m and np.asarray(m["map"]).shape == (1, 24)
