"""Dataset samples of a py: module (list_samples / load_sample): a run by ref, its raw signal and blocks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neurotome import cli
from neurotome.api import create_app
from neurotome.runs import RunManager
from neurotome.store import RunStore

REPO = Path(__file__).resolve().parents[2]
SPEC = "py:examples.signal_model:build"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO))
    mgr = RunManager(RunStore(tmp_path / "runs"))
    return mgr, TestClient(create_app(mgr))


@pytest.fixture()
def run(app):
    mgr, _ = app
    return mgr.create_from_spec(SPEC, None, {"kind": "ref", "ref": "rec02:1"})


def test_list_samples(app):
    _, client = app
    res = client.get("/api/samples", params={"model": SPEC}).json()
    assert res["error"] is None and len(res["samples"]) == 12
    assert res["samples"][0] == {"ref": "rec01:0", "label": "segment 0 · 0–600 s", "group": "rec01"}
    get = lambda spec: client.get("/api/samples", params={"model": spec}).json()  # noqa: E731
    assert get("torchvision:resnet18") == {"samples": [], "error": None}
    assert get("py:examples.custom_model:build")["samples"] == []
    err = client.get("/api/samples", params={"model": "py:no_such_module_xyz:build"}).json()
    assert err["samples"] == [] and "no_such_module_xyz" in err["error"]


def test_run_by_ref_keeps_raw_blocks_time_info(app, run):
    mgr, _ = app
    assert run["sample"] == {"kind": "ref", "ref": "rec02:1", "name": "rec02 · segment 1",
                             "info": {"recording": "rec02", "segment": 1, "start_s": 600, "duration_s": 600,
                                      "fs_hz": 4.0, "pattern": "no dips"}}  # fmt: skip
    assert run["input_shape"] == [1, 74, 10]
    inp = run["input"]
    assert inp["raw"] == {"kind": "signal", "file": "input_raw.npz", "fs": 4.0, "t0": 600.0,
                          "time_label": "s from recording start",
                          "channels": [{"name": "hr", "unit": "bpm", "length": 2400},
                                       {"name": "pressure", "unit": "mmHg", "length": 2400}]}  # fmt: skip
    assert [b["name"] for b in inp["blocks"]] == ["hr · band power", "pressure · band power"]
    assert inp["blocks"][1]["start"] == 5 and inp["blocks"][1]["labels"][0] == "0.0625–0.125 Hz"
    assert inp["time"] == {"axis": 1, "step_s": 8.0, "t0": 608.0}
    assert inp["steps"][0].startswith("Cut segment 1 of rec02") and len(inp["steps"]) == 5
    with np.load(mgr.store.run_dir(run["id"]) / "input_raw.npz") as z:
        assert sorted(z.files) == ["c0", "c1"] and z["c0"].shape == (2400,) and z["c0"].dtype == np.float32
    # OUTPUT_KIND = "values": the top-5 steps by score, no softmax
    out = run["output"]
    assert len(out) == 5 and all(o["prob"] is None for o in out)
    assert [o["logit"] for o in out] == sorted((o["logit"] for o in out), reverse=True)


def test_input_endpoints(app, run):
    _, client = app
    rid = run["id"]
    info = client.get(f"/api/runs/{rid}/input").json()
    assert info["blocks"] == run["input"]["blocks"] and info["time"] == run["input"]["time"]
    assert info["info"]["recording"] == "rec02" and info["raw"]["kind"] == "signal"
    assert [c["name"] for c in info["channels"]] == ["hr · band power", "pressure · band power"]
    assert all(c["min"] <= c["mean"] <= c["max"] for c in info["channels"])
    sig = client.get(f"/api/runs/{rid}/input/signal", params={"points": 100}).json()
    assert sig["fs"] == 4.0 and sig["t0"] == 600.0 and sig["stride"] == 24
    hr = sig["channels"][0]
    assert hr["name"] == "hr" and hr["unit"] == "bpm" and len(hr["mean"]) == len(hr["min"]) == 100
    assert all(lo <= m <= hi for lo, m, hi in zip(hr["min"], hr["mean"], hr["max"], strict=True))
    mean = lambda q: client.get(f"/api/runs/{rid}/input/signal{q}").json()["channels"][1]["mean"]  # noqa: E731
    assert len(mean("")) == 1200 and len(mean("?points=5000")) == 2400  # 1500 points: stride 2
    mat = client.get(f"/api/runs/{rid}/input/matrix").json()
    assert mat["steps"] == 74 and mat["stride"] == 1 and mat["time"]["step_s"] == 8.0
    b = mat["blocks"][1]
    assert b["name"] == "pressure · band power" and len(b["values"]) == 5 and len(b["values"][0]) == 74
    assert b["lo"] <= min(min(r) for r in b["values"]) and len(b["labels"]) == 5
    raw = client.get(f"/api/runs/{rid}/input/raw")
    assert raw.status_code == 200 and raw.headers["content-type"] == "application/octet-stream"


def test_matrix_and_signal_404_for_other_inputs(app):
    mgr, client = app
    rid = mgr.create_from_objects(torch.nn.Linear(4, 2), torch.randn(1, 4))["id"]
    assert client.get(f"/api/runs/{rid}/input/matrix").status_code == 404
    assert client.get(f"/api/runs/{rid}/input/signal").status_code == 404
    seq = mgr.create_from_objects(torch.nn.Linear(4, 2), torch.randn(1, 6, 4))["id"]
    mat = client.get(f"/api/runs/{seq}/input/matrix").json()  # no blocks: one block "input"
    assert [b["name"] for b in mat["blocks"]] == ["input"] and len(mat["blocks"][0]["values"]) == 4


def test_reload_after_restart(app, run):
    mgr, _ = app
    fresh = RunManager(mgr.store)  # a new server over the same runs dir: the model reloads by spec
    live = fresh.live(run["id"])
    assert live.loaded.output_kind == "values" and tuple(live.inputs.shape) == (1, 74, 10)
    res = fresh.attribution(run["id"], "saliency", None, 4)
    assert res["target"] == run["output"][0]["index"]


def test_capture_by_ref_through_the_api_and_cli(app, tmp_path, monkeypatch, capsys):
    _, client = app
    r = client.post("/api/runs", json={"model": SPEC, "sample": {"kind": "ref", "ref": "x" * 513}})
    assert r.status_code == 422
    assert client.post("/api/runs", json={"model": SPEC, "sample": {"kind": "ref"}}).status_code == 400
    monkeypatch.setenv("NEUROTOME_RUNS_DIR", str(tmp_path / "cli"))
    assert cli.main(["capture", SPEC, "--sample", "rec03:2", "--runs-dir", str(tmp_path / "cli")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sample"]["ref"] == "rec03:2" and out["output"][0]["prob"] is None


BAD = '''
import torch
def build():
    return torch.nn.Linear(3, 2)
def load_sample(ref):
    x = torch.zeros(1, 4, 3)
    return {
        "unknown": lambda: 1 / 0,
        "noinput": lambda: {"name": "n"},
        "notdict": lambda: [x],
        "rawshape": lambda: {"input": x, "raw": {"fs": 4, "channels": [{"name": "a", "values": [[1, 2]]}]}},
        "rawfs": lambda: {"input": x, "raw": {"fs": 0, "channels": [{"name": "a", "values": [1]}]}},
        "blocks": lambda: {"input": x, "blocks": [{"name": "b", "start": 0, "stop": 5}]},
        "labels": lambda: {"input": x, "blocks": [{"name": "b", "start": 0, "stop": 3, "labels": ["a"]}]},
        "time": lambda: {"input": x, "time": {"axis": 3, "step_s": 1}},
        "ok": lambda: {"input": x, "info": {"k": [1]}},
    }[ref]()
'''


@pytest.mark.parametrize(("ref", "msg"), [
    ("nope", "failed: KeyError"),
    ("unknown", "failed: ZeroDivisionError"),
    ("noinput", "must return 'input'"),
    ("notdict", "returned list"),
    ("rawshape", "raw channel 0 values must be 1-D"),
    ("rawfs", "positive finite fs"),
    ("blocks", "outside the input's 3 columns"),
    ("labels", "has 1 labels for 3 columns"),
    ("time", "'time' needs an axis"),
])  # fmt: skip
def test_bad_load_sample_is_a_clear_400(app, tmp_path, monkeypatch, ref, msg):
    mgr, _ = app
    (tmp_path / "bad_samples_mod.py").write_text(textwrap.dedent(BAD))
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match=msg):
        mgr.create_from_spec("py:bad_samples_mod:build", None, {"kind": "ref", "ref": ref})
    assert mgr.store.list_runs() == []


def test_minimal_load_sample(app, tmp_path, monkeypatch):
    mgr, client = app
    (tmp_path / "bad_samples_mod.py").write_text(textwrap.dedent(BAD))
    monkeypatch.syspath_prepend(str(tmp_path))
    meta = mgr.create_from_spec("py:bad_samples_mod:build", None, {"kind": "ref", "ref": "ok"})
    assert meta["sample"]["info"] == {"k": "[1]"} and meta["input"]["raw"] is None
    assert meta["input"]["steps"] == ["Custom: bad_samples_mod.load_sample"]
    assert meta["output"][0]["prob"] is not None  # OUTPUT_KIND defaults to logits
    with pytest.raises(ValueError, match="defines no load_sample"):
        mgr.create_from_spec("py:examples.custom_model:build", None, {"kind": "ref", "ref": "a"})
    assert client.get(f"/api/runs/{meta['id']}/input").json()["info"] == {"k": "[1]"}
