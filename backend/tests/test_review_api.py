"""Review regressions: the HTTP boundary (cross-origin writes, uploads, static paths), the job worker, the run
store on Windows, seeded reloads, ``neurotome.open`` called twice and CLI errors."""

from __future__ import annotations

import io
import pathlib
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from test_jobs_m3 import follow

import neurotome
import neurotome.api as api
from neurotome import cli
from neurotome.models import load_model
from neurotome.runs import RunManager
from neurotome.store import RunStore

SPEC = "py:test_pipeline:Tiny"
SAMPLE = {"kind": "random", "shape": [1, 3, 16, 16]}


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    torch.manual_seed(0)
    return RunManager(RunStore(tmp_path / "runs"))


def finish(mgr: RunManager, job) -> dict:
    return follow(mgr.jobs, job.id)[-1]


def test_cross_origin_writes_refused(mgr):
    """A web page may not upload files or start captures (a py: spec runs code) on the local server."""
    c = TestClient(api.create_app(mgr))
    evil = {"Origin": "http://evil.example"}
    files = {"file": ("x.npy", b"data", "application/octet-stream")}
    r = c.post("/api/uploads", files=files, headers=evil)
    assert r.status_code == 403 and r.json()["detail"] == "Cross-origin request refused"
    assert c.post("/api/runs", json={"model": SPEC, "sample": SAMPLE}, headers=evil).status_code == 403
    assert c.delete("/api/runs/run_0001", headers={"Origin": "null"}).status_code == 403
    for ok in ({}, {"Origin": "http://localhost:5173"}, {"Origin": "http://testserver"}):  # testserver = Host
        assert c.post("/api/uploads", files=files, headers=ok).status_code == 200
    assert c.get("/api/runs", headers=evil).status_code == 200  # reads stay open (CORS guards responses)


def test_upload_suffix_never_importable(mgr):
    """An upload is never stored as a module that a py: spec could import from the runs folder."""
    c = TestClient(api.create_app(mgr))
    for name, suffix in (("m.py", ".img"), ("m.pyd", ".img"), ("cat.PNG", ".img"), ("ids.npy", ".npy"),
                         ("x.pt", ".pt"), ("noext", ".img")):  # fmt: skip
        uid = c.post("/api/uploads", files={"file": (name, name.encode())}).json()["upload_id"]
        assert uid.endswith(suffix) and (mgr.uploads / uid).is_file()


def test_uploaded_tensor_is_not_unpickled(mgr, tmp_path):
    marker = tmp_path / "ran.txt"

    class Payload:
        def __reduce__(self):
            return (Path.write_text, (marker, "x"))

    buf = io.BytesIO()
    torch.save({"x": Payload()}, buf)
    uid = mgr.save_upload("evil.pt", buf.getvalue())
    job = finish(mgr, mgr.start_capture(SPEC, None, {"kind": "upload", "upload_id": uid}))
    assert job["status"] == "error" and not marker.exists()


def test_upload_read_is_bounded(mgr, monkeypatch):
    from starlette.datastructures import UploadFile

    sizes, read = [], UploadFile.read

    async def spy(self, size=-1):
        sizes.append(size)
        return await read(self, size)

    monkeypatch.setattr(UploadFile, "read", spy)
    monkeypatch.setattr(api, "MAX_UPLOAD", 10)
    c = TestClient(api.create_app(mgr))
    r = c.post("/api/uploads", files={"file": ("a.npy", b"x" * 100)})
    assert r.status_code == 413 and sizes == [11]
    assert c.post("/api/uploads", files={"file": ("a.npy", b"x" * 10)}).status_code == 200


def test_spa_never_resolves_anchored_paths(mgr, tmp_path, monkeypatch):
    """Resolving \\\\host\\share on Windows contacts that host (NTLM): such paths never reach the disk."""
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<div id=root></div>")
    (static / "app.js").write_text("js")
    monkeypatch.setattr(api, "STATIC_DIR", static)
    seen, resolve = [], pathlib.Path.resolve

    def spy(self, *a, **k):
        seen.append(str(self))
        return resolve(self, *a, **k)

    c = TestClient(api.create_app(mgr))
    monkeypatch.setattr(pathlib.Path, "resolve", spy)
    for url in ("/%5C%5Cneurotome-test.invalid%5Cshare%5Cx", "/%2F%2Fneurotome-test.invalid%2Fshare",
                "/C:%5CWindows%5Cwin.ini"):  # fmt: skip
        r = c.get(url)
        assert r.status_code == 200 and "root" in r.text
    assert not any(s.startswith(("\\\\", "//")) or "win.ini" in s for s in seen)
    assert c.get("/app.js").text == "js"


def test_graph_steps_bounded(mgr):
    rid = mgr.create_from_spec(SPEC, None, SAMPLE)["id"]
    c = TestClient(api.create_app(mgr))
    for steps in (0, 100_000):
        assert c.get(f"/api/runs/{rid}/graph?method=gxa&steps={steps}").status_code == 422
    assert c.get(f"/api/runs/{rid}/graph?method=gxa&steps=8").status_code == 200


def test_job_worker_survives_sys_exit(mgr):
    """A py: module that calls sys.exit (argparse at import) fails its job; the worker keeps running."""
    job = finish(mgr, mgr.start_capture("py:sys:exit", None, SAMPLE))
    assert job["status"] == "error" and "SystemExit" in job["error"]
    assert finish(mgr, mgr.jobs.submit("test", lambda _p: 7))["result"] == 7


def test_delete_during_job_leaves_no_folder(mgr, monkeypatch):
    """A job whose run is deleted before it writes its cache fails instead of recreating the run folder."""
    rid = mgr.create_from_spec(SPEC, None, SAMPLE)["id"]
    deleted, write = threading.Event(), mgr.store.write_json

    def late_write(run_id, name, data):
        if not deleted.is_set():  # the job released its lock: delete the run before it writes
            threading.Thread(target=lambda: (mgr.delete(rid), deleted.set())).start()
            deleted.wait(10)
        return write(run_id, name, data)

    monkeypatch.setattr(mgr.store, "write_json", late_write)
    job = mgr.job(rid, {"kind": "attribution", "method": "gxa"})
    assert finish(mgr, job)["status"] == "error"
    assert not (mgr.store.root / rid).exists() and mgr.store.list_runs() == []


def test_json_writes_survive_concurrent_readers(tmp_path):
    """Windows refuses to replace a file another thread is reading (and to read one being replaced)."""
    store = RunStore(tmp_path)
    store.create("run_0001")
    store.write_json("run_0001", "meta.json", {"a": 0})
    errors, stop = [], time.time() + 1.0

    def loop(fn):
        while time.time() < stop:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    work = [lambda: store.read_json("run_0001", "meta.json")] + [
        lambda: store.write_json("run_0001", "meta.json", {"a": list(range(300))})
    ] * 2
    threads = [threading.Thread(target=loop, args=(f,)) for f in work]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and store.read_json("run_0001", "meta.json") == {"a": list(range(300))}
    assert not list((tmp_path / "run_0001").glob("*.tmp"))


def test_list_runs_numeric_order(tmp_path):
    store = RunStore(tmp_path)
    for n in (9, 9999, 10000):
        rid = f"run_{n:04d}"
        store.create(rid)
        store.write_json(rid, "meta.json", {"id": rid})
    assert [m["id"] for m in store.list_runs()] == ["run_10000", "run_9999", "run_0009"]
    assert store.new_run_id() == "run_10001"


def test_py_spec_reload_same_weights(tmp_path, monkeypatch):
    """A module that draws random numbers at import reloads (LRU eviction, interventions) to the captured
    model: the first load imports it, later loads use the cached module."""
    (tmp_path / "review_rng_mod.py").write_text(
        "import torch\nNOISE = torch.randn(3)\ndef build():\n    return torch.nn.Linear(4, 2)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    a, b = (load_model("py:review_rng_mod:build").model.weight.detach().clone() for _ in range(2))
    assert torch.equal(a, b)


def test_open_twice_reuses_server(tmp_path, monkeypatch):
    """A second neurotome.open (notebook, block=False) adds its run to the running server, whose manager then
    holds the new model; a new server could not bind the port."""
    import uvicorn
    from test_pipeline import Tiny

    monkeypatch.setenv("NEUROTOME_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(neurotome, "_servers", {})
    stop, servers = threading.Event(), []

    class FakeServer:
        def __init__(self, config):
            servers.append(config.app)

        def run(self):
            stop.wait(10)

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    try:
        u1 = neurotome.open(Tiny(), torch.randn(1, 3, 16, 16), port=1, block=False)
        u2 = neurotome.open(Tiny(), torch.randn(1, 3, 16, 16), port=1, block=False)
        assert len(servers) == 1
        ids = [u.split("/runs/")[1].split("/")[0] for u in (u1, u2)]
        c = TestClient(servers[0])
        assert c.post(f"/api/runs/{ids[1]}/attribution", json={"method": "gxa"}).status_code == 200
    finally:
        stop.set()


def test_cli_capture_error_is_plain(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    monkeypatch.setenv("NEUROTOME_RUNS_DIR", str(tmp_path))
    assert cli.main(["capture", SPEC, "--text", "hi", "--runs-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "This model has no tokenizer" in err and "Traceback" not in err
    assert RunStore(tmp_path).list_runs() == []
