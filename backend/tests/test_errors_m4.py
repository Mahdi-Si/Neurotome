"""M4 errors: every failure reaches the user as a specific message with the right HTTP status."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from test_jobs_m3 import follow
from test_pipeline import Tiny
from torch import nn

import neurotome.runs
from neurotome.api import create_app
from neurotome.errors import NO_GRAD, TraceError, describe
from neurotome.models import LoadedModel, to_logits
from neurotome.runs import RunManager
from neurotome.store import RunStore

INPUT_SHAPE = [1, 4]  # py:test_errors_m4:<Name> models take (1, 4) inputs unless a sample says otherwise
SAMPLE = {"kind": "random", "shape": [1, 4]}


class Boom(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        raise RuntimeError("boom inside forward")


class Namespace(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return SimpleNamespace(score=self.fc(x))


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4, 3)

    def forward(self, ids):
        return self.fc(self.emb(ids).mean(1))


class Detached(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return torch.from_numpy(self.fc(x).detach().numpy()) * 2


class DetachedMiddle(Detached):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(3, 2)

    def forward(self, x):
        return self.out(super().forward(x))


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    torch.manual_seed(0)
    return RunManager(RunStore(tmp_path / "runs"))


def run_dirs(mgr: RunManager) -> list[str]:
    return [p.name for p in mgr.store.root.glob("run_*")]


def finish(mgr: RunManager, job) -> dict:
    return follow(mgr.jobs, job.id)[-1]


def test_trace_failure(mgr):
    with pytest.raises(TraceError, match="could not trace.*boom inside forward") as e:
        mgr.create_from_objects(Boom(), torch.randn(1, 4))
    assert describe(e.value)[0] == 422 and run_dirs(mgr) == []
    body = {"model": "py:test_errors_m4:Boom", "sample": SAMPLE}
    r = TestClient(create_app(mgr)).post("/api/runs", json=body)
    job = finish(mgr, mgr.jobs.get(r.json()["id"]))
    assert job["status"] == "error" and "TorchLens could not trace forward()" in job["error"]
    assert "boom inside forward" in job["error"] and run_dirs(mgr) == []
    load = finish(mgr, mgr.start_capture("py:nope_missing_xyz:build", None, SAMPLE))
    assert load["error"].startswith("Cannot load model 'py:nope_missing_xyz:build': ModuleNotFoundError")


def test_non_tensor_output(mgr):
    meta = mgr.create_from_objects(Namespace(), torch.randn(1, 4))
    assert meta["output"] == [] and "SimpleNamespace" in meta["output_error"]
    c = TestClient(create_app(mgr))
    r = c.post(f"/api/runs/{meta['id']}/attribution", json={"method": "gxa"})
    assert r.status_code == 400 and "returned SimpleNamespace" in r.json()["detail"]
    job = c.post(f"/api/runs/{meta['id']}/jobs", json={"kind": "attribution", "method": "gxa"}).json()
    assert finish(mgr, mgr.jobs.get(job["id"]))["error"] == r.json()["detail"]

    t = torch.randn(2, 5)
    hf = type("HF", (dict,), {"logits": None})(x=t)  # ModelOutput-like: .logits None -> the dict value
    for out in ({"a": 1, "logits": t}, (None, t), [t], SimpleNamespace(logits=t), hf, torch.tensor(3.0),
                torch.randn(2, 4, 5)):  # fmt: skip
        got = to_logits(out)
        assert got.dim() == 2 and got.is_floating_point()
    assert to_logits(torch.tensor(3.0)).shape == (1, 1) and to_logits(torch.randn(2, 4, 5)).shape == (2, 5)
    for bad in ("x", {}, torch.tensor([1, 2])):
        with pytest.raises(ValueError):
            to_logits(bad)


def test_integer_input(mgr):
    meta = mgr.create_from_objects(Embed(), torch.randint(0, 10, (1, 5)))
    res = mgr.attribution(meta["id"], "ig", None, 8)
    assert res["input"] is None and "needs a float input" in res["input_note"] and res["scores"]
    op = next(o["id"] for o in mgr.store.ir(meta["id"])["ops"] if o["type"] == "linear")
    job = finish(mgr, mgr.job(meta["id"], {"kind": "layer", "method": "layer_ig", "op": op}))
    assert job["status"] == "error" and "need a float input" in job["error"]
    assert mgr.attribution(mgr.create_from_objects(Small(), torch.randn(1, 4))["id"], "gxa", None, 8)[
        "input_note"] is None  # fmt: skip

    buf = io.BytesIO()
    np.save(buf, np.array([[1, 2, 3]], dtype=np.int32))
    uid = mgr.save_upload("ids.npy", buf.getvalue())
    x, _ = mgr.build_input(LoadedModel(Embed(), None, None), {"kind": "upload", "upload_id": uid})
    assert x.dtype == torch.int64


def test_no_gradient(mgr):
    c = TestClient(create_app(mgr))
    for model in (Detached(), DetachedMiddle()):  # no grad at all; params after the break but none before
        meta = mgr.create_from_objects(model, torch.randn(1, 4))
        r = c.post(f"/api/runs/{meta['id']}/attribution", json={"method": "gxa"})
        assert r.status_code == 400 and r.json()["detail"] == NO_GRAD
    e = RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    assert describe(e) == (400, NO_GRAD)


class ParamFree(nn.Module):
    def forward(self, x):
        return torch.relu(x).flatten(1) * 2


def test_gradients_without_trainable_params(mgr):
    frozen = Small().requires_grad_(False)
    for model in (ParamFree(), frozen):  # the input leaf carries the gradient
        res = mgr.attribution(mgr.create_from_objects(model, torch.randn(1, 4))["id"], "gxa", None, 8)
        assert any(v != 0 for v in res["scores"].values())


def test_missing_tensor_details(mgr):
    c = TestClient(create_app(mgr))
    x = torch.randn(1, 3, 16, 16)
    run = mgr.create_from_objects(Tiny(), x, max_capture_mb=(3 * 256 * 4 + 20 + 100) / 2**20)["id"]
    skipped = next(o["id"] for o in mgr.store.ir(run)["ops"] if o["unsaved"] == "budget")
    r = c.get(f"/api/runs/{run}/ops/{skipped}/tensor")
    assert r.status_code == 404 and "capture budget (" in r.json()["detail"]
    assert r.json()["detail"].startswith(f"No tensor saved for {skipped}: ")
    sel = mgr.create_from_objects(Tiny(), x, modules=["block"])["id"]
    stem = next(o["id"] for o in mgr.store.ir(sel)["ops"] if o["unsaved"] == "modules")
    r = c.get(f"/api/runs/{sel}/ops/{stem}/slice")
    assert r.status_code == 404 and "outside the captured modules (block)" in r.json()["detail"]
    r = c.get(f"/api/runs/{sel}/diff", params={"a": stem, "b": "nope"})
    assert r.status_code == 404 and "outside the captured modules" in r.json()["detail"]
    r = c.get(f"/api/runs/{sel}/ops/nope/tensor")
    assert r.status_code == 404 and r.json()["detail"] == f"Op nope not found in {sel}"


def test_model_can_no_longer_be_loaded(mgr):
    spec_run = mgr.create_from_spec("py:test_errors_m4:Small", None, SAMPLE)["id"]
    proc_run = mgr.create_from_objects(Small(), torch.randn(1, 4))["id"]
    fresh = RunManager(mgr.store)
    meta = fresh.store.meta(spec_run)
    meta["model"]["spec"] = "py:gone_module_xyz:build"
    fresh.store.write_json(spec_run, "meta.json", meta)
    c = TestClient(create_app(fresh))
    r = c.post(f"/api/runs/{spec_run}/attribution", json={"method": "gxa"})
    assert r.status_code == 409 and "can no longer be loaded" in r.json()["detail"]
    assert "gone_module_xyz" in r.json()["detail"]
    r = c.post(f"/api/runs/{proc_run}/attribution", json={"method": "gxa"})
    assert r.status_code == 409 and "in-process" in r.json()["detail"]


def test_cuda_oom(mgr, monkeypatch):
    def oom(*_a, **_k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(neurotome.runs, "capture", oom)
    job = finish(mgr, mgr.start_capture("py:test_errors_m4:Small", None, SAMPLE))
    assert job["status"] == "error" and "out of memory" in job["error"] and "--device cpu" in job["error"]
    assert run_dirs(mgr) == []
    assert describe(torch.cuda.OutOfMemoryError("CUDA out of memory"))[0] == 413


def test_describe_fallbacks():
    assert describe(RuntimeError("x")) == (500, "RuntimeError: x")
    assert describe(ValueError("")) == (400, "ValueError")
    assert describe(KeyError("k")) == (500, "KeyError: 'k'")


def test_spa_fallback_guards(mgr, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import neurotome.api as api

    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<div id=root></div>")
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setattr(api, "STATIC_DIR", static)
    c = TestClient(api.create_app(mgr))
    r = c.get("/api/nope")
    assert r.status_code == 404 and "No API route" in r.json()["detail"]
    assert "root" in c.get("/runs/x/graph").text
    assert "secret" not in c.get("/..%2Fsecret.txt").text
