"""M3 jobs: the in-process runner, the job HTTP routes and the SSE stream."""

from __future__ import annotations

import json
import threading
import time

import pytest
import torch
from test_pipeline import Tiny

from neurotome.jobs import TERMINAL, JobRunner
from neurotome.runs import RunManager
from neurotome.store import RunStore


def follow(runner, job_id, timeout=60.0):
    """Every state seen through wait() until the job is terminal."""
    seen, v = [], -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        data, v = runner.wait(job_id, v, 5.0)
        seen.append(data)
        if data["status"] in TERMINAL:
            return seen
    raise AssertionError("job did not finish")


def test_runner_lifecycle():
    r = JobRunner()

    def work(progress):
        for f in (0.25, 0.5, 0.75):
            progress(f, f"step {f}")
            time.sleep(0.02)
        return 42

    job = r.submit("test", work)
    seen = follow(r, job.id)
    assert seen[-1]["status"] == "done" and seen[-1]["result"] == 42 and seen[-1]["progress"] == 1
    fracs = [s["progress"] for s in seen]
    assert fracs == sorted(fracs)
    keys = {"id", "kind", "run_id", "status", "progress", "message", "result", "error", "created"}
    assert set(seen[-1]) == keys

    def boom(_p):
        raise RuntimeError("bad input")

    failed = follow(r, r.submit("test", boom).id)[-1]
    assert failed["status"] == "error" and failed["error"] == "bad input"
    assert follow(r, r.submit("test", lambda _p: 1 / 0).id)[-1]["error"] == "division by zero"
    with pytest.raises(KeyError):
        r.get("job_999999")


def test_cancel_and_dedup():
    r = JobRunner()
    gate, ran = threading.Event(), []
    blocker = r.submit("test", lambda _p: gate.wait(10))
    queued = r.submit("test", lambda _p: ran.append(1), key="k")
    assert r.submit("test", lambda _p: None, key="k").id == queued.id  # same key while queued
    assert r.cancel(queued.id).status == "cancelled"
    gate.set()
    assert follow(r, blocker.id)[-1]["status"] == "done"

    def loop(progress):
        while True:
            progress(0.1, "looping")
            time.sleep(0.01)

    running = r.submit("test", loop, key="loop")
    while r.get(running.id).status != "running":
        time.sleep(0.01)
    r.cancel(running.id)
    again = r.submit("test", lambda _p: "fresh", key="loop")  # a cancelling job is never reused
    assert again.id != running.id
    assert follow(r, running.id)[-1]["status"] == "cancelled"
    assert follow(r, again.id)[-1]["result"] == "fresh"
    assert not ran
    done = r.finished("test", {"x": 1}, "run_0001")
    assert done.status == "done" and done.result == {"x": 1} and done.progress == 1


def test_runner_keeps_newest():
    r = JobRunner(keep=3)
    ids = [r.finished("test", i).id for i in range(5)]
    for jid in ids[:2]:
        with pytest.raises(KeyError):
            r.get(jid)
    assert [r.get(j).result for j in ids[2:]] == [2, 3, 4]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from neurotome.api import create_app

    torch.manual_seed(0)
    mgr = RunManager(RunStore(tmp_path_factory.mktemp("jobs") / "runs"))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), name="tiny")
    return TestClient(create_app(mgr)), run["id"]


def poll(c, job, timeout=60.0):
    deadline = time.time() + timeout
    while job["status"] not in TERMINAL and time.time() < deadline:
        time.sleep(0.05)
        job = c.get(f"/api/jobs/{job['id']}").json()
    return job


def test_http_jobs(client):
    c, run = client
    r = c.post(f"/api/runs/{run}/jobs", json={"kind": "attribution", "method": "gxa"})
    assert r.status_code == 200 and r.json()["kind"] == "attribution" and r.json()["run_id"] == run
    done = poll(c, r.json())
    assert done["status"] == "done" and "merges" in done["result"]
    again = c.post(f"/api/runs/{run}/jobs", json={"kind": "attribution", "method": "gxa"}).json()
    assert again["status"] == "done" and again["id"] != done["id"]
    assert c.post("/api/runs/run_9999/jobs", json={"kind": "attribution"}).status_code == 404
    assert c.post(f"/api/runs/{run}/jobs", json={"kind": "layer"}).status_code == 400
    assert c.post(f"/api/runs/{run}/jobs", json={"kind": "attribution", "method": "x"}).status_code == 400
    assert c.post(f"/api/runs/{run}/jobs", json={"kind": "nope"}).status_code == 422
    jobs = f"/api/runs/{run}/jobs"
    assert c.post(jobs, json={"kind": "neuron", "op": "a", "channel": -1}).status_code == 422
    assert c.get("/api/jobs/job_999999").status_code == 404
    assert c.post("/api/jobs/job_999999/cancel").status_code == 404
    assert c.get("/api/jobs/job_999999/events").status_code == 404
    assert c.post(f"/api/jobs/{done['id']}/cancel").json()["status"] == "done"


def test_sse_stream(client):
    c, run = client
    job = c.post(f"/api/runs/{run}/jobs", json={"kind": "layer", "op": "conv2d_2_3", "steps": 8}).json()
    events = []
    with c.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events and all(e["id"] == job["id"] for e in events)
    assert events[-1]["status"] == "done" and len(events[-1]["result"]["channels"]) == 8


def test_http_objective(client):
    c, run = client
    url = f"/api/runs/{run}"
    g = c.get(f"{url}/graph", params={"depth": 3, "method": "gxa", "obj_op": "relu_2_4", "obj_ch": 3}).json()
    assert g["path"][0] == "relu_2_4"
    assert not any(e.get("share_method") for e in g["edges"])  # the add lies after the unit: no split
    g = c.get(f"{url}/graph", params={"depth": 3, "method": "gxa"}).json()
    assert g["path"][0] == "output_1" and any(e.get("share_method") == "merge" for e in g["edges"])
    assert c.get(f"{url}/graph", params={"method": "gxa", "obj_op": "relu_2_4"}).status_code == 400
    assert c.get(f"{url}/graph", params={"method": "bad"}).status_code == 400
    body = {"method": "gxa", "objective": {"op": "relu_2_4", "channel": 3}}
    a = c.post(f"{url}/attribution", json=body).json()
    assert a["objective"]["kind"] == "unit" and a["prob"] is None
    body["objective"]["op"] = "add_1_6"
    r = c.post(f"{url}/attribution", json=body)
    assert r.status_code == 400 and "not the output of a module" in r.json()["detail"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_jobs_on_cuda(tmp_path):
    mgr = RunManager(RunStore(tmp_path / "runs"), device="cuda")
    run = mgr.create_from_objects(Tiny().cuda(), torch.randn(1, 3, 16, 16))["id"]
    iv = {"kind": "intervene", "interventions": {"edits": [{"op": "conv2d_3_5", "kind": "zero"}]}}
    for req in ({"kind": "layer", "op": "conv2d_2_3", "steps": 8}, iv):
        seen = follow(mgr.jobs, mgr.job(run, req).id, timeout=120)
        assert seen[-1]["status"] == "done", seen[-1]["error"]


def test_sse_non_finite(client):
    """The stream encodes NaN/Inf as null, like the JSON replies (JSON.parse rejects bare NaN)."""
    c, _ = client
    job = c.app.state.manager.jobs.submit("test", lambda _p: {"x": float("nan"), "y": [float("inf"), 1.0]})
    with c.stream("GET", f"/api/jobs/{job.id}/events") as r:
        data = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    last = json.loads(data[-1], parse_constant=lambda k: pytest.fail(f"bare {k} in SSE"))
    assert last["result"] == {"x": None, "y": [None, 1.0]} == c.get(f"/api/jobs/{job.id}").json()["result"]
