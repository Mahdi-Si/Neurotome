"""M4 capture: capture jobs, cancellation, the never-cross budget, selective capture, CLI flags, old IRs."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest
import torch
from test_jobs_m3 import follow
from test_pipeline import Tiny

from neurotome import cli
from neurotome.graph import graph_view
from neurotome.runs import RunManager
from neurotome.store import RunStore

SPEC = "py:test_pipeline:Tiny"
SAMPLE = {"kind": "random", "shape": [1, 3, 16, 16]}
MB = 1024 * 1024


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    torch.manual_seed(0)
    return RunManager(RunStore(tmp_path / "runs"))


def run_dirs(mgr: RunManager) -> list[str]:
    return [p.name for p in mgr.store.root.glob("run_*")]


def test_capture_job_reports_progress(mgr, monkeypatch):
    calls, submit = [], mgr.jobs.submit  # every progress call (wait() may coalesce updates)

    def recording(kind, fn, *args, **kw):
        return submit(kind, lambda p: fn(lambda f, m: (calls.append((f, m)), p(f, m))), *args, **kw)

    monkeypatch.setattr(mgr.jobs, "submit", recording)
    seen = follow(mgr.jobs, mgr.start_capture(SPEC, None, SAMPLE).id)
    assert seen[-1]["status"] == "done" and seen[-1]["kind"] == "capture" and seen[-1]["run_id"] is None
    messages = [m for _, m in calls]
    assert messages[:3] == ["Load model", "Build input", "Trace"] and messages[-1] == "Summarise"
    saves = [re.fullmatch(r"Save tensors (\d+)/(\d+)", m) for m in messages[3:-1]]
    last = saves[-1]
    assert all(saves) and last and last[1] == last[2] == "12"
    for fracs in ([f for f, _ in calls], [s["progress"] for s in seen]):
        assert fracs == sorted(fracs)
    meta = seen[-1]["result"]
    assert meta["id"] in [r["id"] for r in mgr.store.list_runs()]


def test_cancel_capture_queued_and_running(mgr, monkeypatch):
    gate = threading.Event()
    blocker = mgr.jobs.submit("test", lambda _p: gate.wait(10))
    queued = mgr.start_capture(SPEC, None, SAMPLE)
    assert mgr.jobs.cancel(queued.id).status == "cancelled"
    gate.set()
    follow(mgr.jobs, blocker.id)
    assert run_dirs(mgr) == []

    build = RunManager.build_input

    def slow(self, loaded, sample):
        time.sleep(0.5)
        return build(self, loaded, sample)

    monkeypatch.setattr(RunManager, "build_input", slow)
    job = mgr.start_capture(SPEC, None, SAMPLE)
    deadline = time.time() + 30
    while mgr.jobs.get(job.id).message != "Build input" and time.time() < deadline:
        time.sleep(0.01)
    mgr.jobs.cancel(job.id)
    assert follow(mgr.jobs, job.id)[-1]["status"] == "cancelled"
    assert run_dirs(mgr) == []


def test_budget_never_crossed(mgr):
    x = torch.randn(1, 3, 16, 16)
    budget = 3 * 16 * 16 * 4 + 5 * 4 + 2 * 8 * 16 * 16 * 4 + 100  # input, output and two conv-sized ops
    meta = mgr.create_from_objects(Tiny(), x, max_capture_mb=budget / MB)
    ir = mgr.store.ir(meta["id"])
    model, ops = ir["model"], ir["ops"]
    assert model["truncated"] and model["budget_bytes"] == int(budget / MB * MB)
    saved = [o for o in ops if o["saved"]]
    assert sum(4 * torch.Size(o["shape"]).numel() for o in saved) <= model["budget_bytes"]
    assert ops[0]["saved"] and ops[-1]["saved"] and ops[0]["unsaved"] is None
    body = ops[1:-1]
    first = next(i for i, o in enumerate(body) if not o["saved"])
    assert first == 2 and all(o["unsaved"] == "budget" and o["stats"] for o in body[first:])
    assert model["saved_ops"] == sum(o["saved"] for o in body) == 2
    assert meta["model"]["saved_ops"] == 2 and meta["output_error"] is None
    nodes = graph_view(ir, 3)["nodes"]
    assert all("unsaved" in n for n in nodes) and any(n["unsaved"] == "budget" for n in nodes)

    big = mgr.store.ir(mgr.create_from_objects(Tiny(), x, max_capture_mb=64)["id"])
    assert not big["model"]["truncated"] and all(o["unsaved"] is None for o in big["ops"])


def test_selective_capture(mgr):
    x = torch.randn(1, 3, 16, 16)
    full = mgr.store.ir(mgr.create_from_objects(Tiny(), x)["id"])
    ir = mgr.store.ir(mgr.create_from_objects(Tiny(), x, modules=[" block ", ""])["id"])
    assert len(ir["ops"]) == len(full["ops"]) and ir["model"]["capture_modules"] == ["block"]
    for o in ir["ops"]:
        inside = any(c.startswith("block") for c in o["modules"])
        if o["is_input"] or o["is_output"] or inside:
            assert o["saved"] and o["unsaved"] is None
        else:
            assert o["unsaved"] == "modules" and o["stats"]
    assert ir["model"]["saved_ops"] == 5 and not ir["model"]["truncated"]
    before = run_dirs(mgr)
    with pytest.raises(ValueError, match=r"Unknown module\(s\) nope; .*e\.g\. stem, block"):
        mgr.create_from_objects(Tiny(), x, modules=["nope"])
    assert run_dirs(mgr) == before


def test_http_capture(mgr):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    c = TestClient(create_app(mgr))
    body = {"model": SPEC, "sample": SAMPLE, "modules": ["block"], "max_capture_mb": 64}
    r = c.post("/api/runs", json=body)
    assert r.status_code == 202 and r.json()["kind"] == "capture" and r.json()["run_id"] is None
    job = r.json()
    deadline = time.time() + 60
    while job["status"] not in ("done", "error", "cancelled") and time.time() < deadline:
        time.sleep(0.05)
        job = c.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "done" and job["result"]["model"]["capture_modules"] == ["block"]
    assert job["result"]["model"]["budget_bytes"] == 64 * MB
    assert c.post("/api/runs", json={"model": "resnet18"}).status_code == 400
    assert c.post("/api/runs", json={"model": SPEC, "max_capture_mb": 0}).status_code == 422
    inf = c.post(
        "/api/runs",
        content=f'{{"model": "{SPEC}", "max_capture_mb": Infinity}}',
        headers={"content-type": "application/json"},
    )
    assert inf.status_code in (400, 422)  # rejected before a job starts (the 422 body cannot echo inf)
    with pytest.raises(ValueError, match="positive finite"):
        mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), max_capture_mb=float("inf"))
    missing = c.post("/api/runs", json={"model": SPEC, "sample": {"kind": "upload", "upload_id": "nope.npy"}})
    assert missing.status_code == 400 and "upload the file again" in missing.json()["detail"]
    health = c.get("/api/health").json()
    assert health["version"] == "0.1.0" and health["max_capture_mb"] is None
    mgr.max_capture_bytes = 512 * MB
    assert c.get("/api/health").json()["max_capture_mb"] == 512


def test_cli_capture_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    monkeypatch.setenv("NEUROTOME_RUNS_DIR", str(tmp_path))  # restored after the CLI overwrites it
    assert cli.main(["capture", SPEC, "--modules", "block, pool", "--max-capture-mb", "8",
                     "--runs-dir", str(tmp_path)]) == 0  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    ir = RunStore(tmp_path).ir(out["id"])
    assert ir["model"]["capture_modules"] == ["block", "pool"] and ir["model"]["budget_bytes"] == 8 * MB


def test_old_ir_derives_unsaved(mgr):
    ir = mgr.store.ir(mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))["id"])
    for o in ir["ops"]:
        del o["unsaved"]
    for k in ("saved_ops", "budget_bytes", "capture_modules"):
        del ir["model"][k]
    relu = next(o for o in ir["ops"] if o["id"] == "relu_3_7")
    relu["saved"] = False
    nodes = {n["output_op"]: n for n in graph_view(ir, 3)["nodes"]}
    assert nodes["relu_3_7"]["unsaved"] == "no_tensor"
    assert all(n["unsaved"] is None for op, n in nodes.items() if op != "relu_3_7")
    ir["model"]["truncated"] = True
    assert {n["output_op"]: n for n in graph_view(ir, 3)["nodes"]}["relu_3_7"]["unsaved"] == "budget"
