"""Step selection: the whole output (GET /output), a module's target_windows hook (GET /windows?t=) and the
unpooled series map it is drawn on."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neurotome.api import create_app
from neurotome.attribution import input_map
from neurotome.runs import INPUT_MAP, OUTPUT, RunManager, check_windows
from neurotome.store import RunStore

REPO = Path(__file__).resolve().parents[2]
SPEC = "py:examples.signal_model:build"


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(REPO))
        mgr = RunManager(RunStore(tmp_path_factory.mktemp("win") / "runs"))
        run = mgr.create_from_spec(SPEC, None, {"kind": "ref", "ref": "rec02:1"})
        yield mgr, TestClient(create_app(mgr)), run


def test_windows_of_a_step(app):
    _, client, run = app
    get = lambda t: client.get(f"/api/runs/{run['id']}/windows", params={"t": t}).json()  # noqa: E731
    res = get(40)
    assert res["error"] is None
    assert res["windows"] == [
        {"label": "step t40", "start": 40, "stop": 41, "blocks": None, "kind": "target"},
        {"label": "pressure context (12 steps)", "start": 28, "stop": 40, "blocks": ["pressure · band power"],
         "kind": "context"},
        {"label": "horizon t+1…t+3", "start": 41, "stop": 44, "blocks": None, "kind": "horizon"},
    ]
    # clipped to [0, 74): the context of step 5 starts at 0; step 73 has no horizon left (dropped)
    assert get(5)["windows"][1]["start"] == 0
    assert [w["kind"] for w in get(73)["windows"]] == ["target", "context"]
    assert client.get(f"/api/runs/{run['id']}/windows", params={"t": -1}).status_code == 422
    assert client.get("/api/runs/run_nope/windows", params={"t": 0}).status_code == 404


def test_bad_hook_output_is_an_error_not_a_crash(app, monkeypatch):
    _, client, run = app
    mod = importlib.import_module("examples.signal_model")
    get = lambda: client.get(f"/api/runs/{run['id']}/windows", params={"t": 3}).json()  # noqa: E731
    w = {"start": 0, "stop": 4, "kind": "context", "blocks": None}
    for got, why in [
        ({"start": 0}, "must return a list of dicts, got dict"),
        ([w, 7], "window 1 is int, not a dict"),
        ([{**w, "stop": 4.5}], "needs int start and stop"),
        ([{**w, "kind": "past"}], "has kind 'past'"),
        ([{**w, "blocks": ["fhr_st"]}], "unknown block 'fhr_st'; the input's blocks are 'hr · band power'"),
    ]:
        monkeypatch.setattr(mod, "target_windows", lambda _i, got=got: got)
        res = get()
        assert res["windows"] == [] and why in res["error"], (got, res)

    def boom(i):
        raise KeyError(i)

    monkeypatch.setattr(mod, "target_windows", boom)
    assert get() == {"windows": [], "error": "target_windows(3) failed: KeyError: 3"}
    monkeypatch.delattr(mod, "target_windows")  # no hook: no windows, no error
    assert get() == {"windows": [], "error": None}


def test_default_block_names_and_time_axis():
    # a raw (1, C, T) input: time on the longer axis, one block per raw channel
    raw = {"kind": "signal", "channels": [{"name": "fhr"}, {"name": "up"}]}
    got = [{"start": 5000, "stop": 6000, "kind": "horizon", "blocks": ["up"]}]
    assert check_windows(got, [1, 2, 5280], {"raw": raw}) == [
        {"label": "horizon 5000–5279", "start": 5000, "stop": 5280, "blocks": ["up"], "kind": "horizon"}]
    # nothing to name the channels by: one block "input"
    assert check_windows([{"start": 0, "stop": 1, "kind": "target", "blocks": ["input"]}], [1, 300, 126], {})
    with pytest.raises(ValueError, match="unknown block 'up'"):
        check_windows(got, [1, 300, 126], {})
    assert check_windows([], None, {}) == []


def test_output_is_every_step_with_its_label(app):
    mgr, client, run = app
    out = client.get(f"/api/runs/{run['id']}/output").json()
    assert out["n"] == 74 and out["stride"] == 1 and len(out["values"]) == 74
    assert out["labels"][0] == "t0 · +0:08" and out["labels"][73] == "t73 · +9:52"
    top = run["output"][0]
    assert out["values"][top["index"]] == pytest.approx(top["logit"], rel=1e-4)
    # a run captured before output.npy existed: one forward pass of the live model, then the file
    path = mgr.store.run_dir(run["id"]) / OUTPUT
    saved = np.load(path)
    path.unlink()
    assert client.get(f"/api/runs/{run['id']}/output").json()["values"] == out["values"]
    assert np.allclose(np.load(path), saved)


def test_series_map_unpooled_up_to_512_steps_and_restamped(app):
    m = input_map(torch.randn(1, 512, 3), None)
    assert m["stride"] == 1 and np.asarray(m["map"]).shape == (3, 512)
    assert np.asarray(m["map"]).sum() * m["scale"] == pytest.approx(m["total"], rel=1e-2)
    assert input_map(torch.randn(1, 513, 3), None)["stride"] == 2
    # an attribution cached under the old stamp (time pooled to 128 columns) is recomputed
    mgr, _, run = app
    res = mgr.attribution(run["id"], "gxa", 7, 4)
    name = "attribution/gxa_7_4.json"
    mgr.store.write_json(run["id"], name, {**res, "input_map": "series"})
    assert mgr._cached(run["id"], name) is None
    assert mgr.attribution(run["id"], "gxa", 7, 4)["input_map"] == INPUT_MAP
