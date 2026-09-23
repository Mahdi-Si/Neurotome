"""HTTP API. Everything the frontend needs, JSON only; tensors leave the server as small summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter

from . import __version__
from .attribution import METHODS
from .config import get_settings
from .errors import describe
from .graph import graph_view, module_tree, search_index, unsaved
from .jobs import TERMINAL, Job
from .models import SUGGESTED_MODELS, TEXT_MODELS
from .runs import RunManager, _module
from .slicing import SliceTooLarge, slice_npy, slice_table
from .store import PREVIEW, safe_name
from .tensors import tensor_diff, tensor_view

STATIC_DIR = Path(__file__).parent / "static"
_JSON = TypeAdapter(Any)


class Sample(BaseModel):
    kind: str = Field(default="random", pattern="^(random|upload|text)$")
    upload_id: str | None = None
    text: str | None = Field(default=None, max_length=10000)  # kind text
    name: str | None = None
    shape: list[int] | None = None
    seed: int = 0


class CreateRun(BaseModel):
    model: str = "torchvision:resnet18"
    weights: str | None = None
    sample: Sample = Sample()
    modules: list[str] | None = None  # module addresses; None or [] = every op
    # per-run override of NEUROTOME_MAX_CAPTURE_MB
    max_capture_mb: float | None = Field(None, gt=0, allow_inf_nan=False)
    backend: Literal["torchlens", "nnsight"] | None = None  # None = auto (NNsight above nnsight_params)


class Objective(BaseModel):
    op: str
    channel: int = Field(ge=0)


class AttributionRequest(BaseModel):
    method: str = "ig"
    target: int | None = None
    steps: int = Field(32, ge=1, le=512)
    objective: Objective | None = None


class JobRequest(BaseModel):
    kind: Literal["attribution", "layer", "neuron", "agreement", "patch", "lens"]
    method: str | None = None  # attribution: gxa|saliency|ig (default ig); layer: layer_ig|layer_cond
    target: int | None = None  # lens: a token id, None = the top prediction
    objective: Objective | None = None
    steps: int = Field(32, ge=1, le=512)
    op: str | None = None  # layer / neuron / patch; agreement: optional (units block)
    channel: int | None = Field(None, ge=0)  # neuron
    baseline: Literal["zeros", "mean", "run"] = "zeros"  # patch
    source_run: str | None = None  # patch, baseline == "run"


def create_app(manager: RunManager | None = None) -> FastAPI:
    settings = get_settings()
    if manager is None:
        manager = RunManager.from_settings(settings)
    store = manager.store
    app = FastAPI(title="Neurotome", version=__version__)
    app.state.manager = manager

    async def error(_request: Request, e: Exception) -> JSONResponse:
        status, msg = describe(e)
        return JSONResponse({"detail": msg}, status)

    # not Exception: Starlette re-raises from that handler, which breaks TestClient
    app.add_exception_handler(ValueError, error)
    app.add_exception_handler(RuntimeError, error)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _meta(run_id: str) -> dict[str, Any]:
        try:
            return store.meta(run_id)
        except KeyError:
            raise HTTPException(404, f"Run {run_id} not found") from None

    def _ir(run_id: str) -> dict[str, Any]:
        _meta(run_id)
        return store.ir(run_id)

    def _job(job_id: str) -> Job:
        try:
            return manager.jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"Job {job_id} not found") from None

    def _tensor(run_id: str, op_id: str):
        _meta(run_id)
        try:
            return store.load_tensor(run_id, op_id)
        except KeyError:
            raise HTTPException(404, _no_tensor(run_id, op_id)) from None

    def _no_tensor(run_id: str, op_id: str) -> str:
        ir = store.ir(run_id)
        op = next((o for o in ir["ops"] if o["id"] == op_id), None)
        if op is None:
            return f"Op {op_id} not found in {run_id}"
        why, model = unsaved(ir, op), ir["model"]
        if why == "budget":
            budget = model.get("budget_bytes") or manager.max_capture_bytes or 0
            reason = (f"the capture budget ({budget / 2**20:g} MB) was reached before this op; capture again "
                      "with a larger budget")  # fmt: skip
        elif why == "modules":
            mods = ", ".join(model.get("capture_modules") or [])
            reason = f"it is outside the captured modules ({mods}); capture again with its module selected"
        else:
            reason = "the op produced no tensor"
        return f"No tensor saved for {op_id}: {reason}"

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        import torch

        return {
            "ok": True,
            "device": manager.device,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "runs_dir": str(store.root),
            "version": __version__,
            "max_capture_mb": manager.max_capture_bytes / 2**20 if manager.max_capture_bytes else None,
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {
            "suggested": SUGGESTED_MODELS,
            "methods": METHODS,
            "text": TEXT_MODELS,
            "backends": {
                "hf": importlib.util.find_spec("transformers") is not None,
                "nnsight": _module("nnsight").available(),
            },
            "nnsight_params": manager.nnsight_params,
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return store.list_runs()

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:
        data = await file.read()
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(413, "Upload larger than 64 MB")
        name = file.filename or ""
        return {"upload_id": manager.save_upload(name or "upload", data), "name": name}

    @app.post("/api/runs", status_code=202)
    def create_run(req: CreateRun) -> dict[str, Any]:
        """Start a capture job; its result is the run meta. 400 for a bad spec prefix, a missing upload, an
        empty text sample or the nnsight backend without its package."""
        sample = req.sample.model_dump()
        job = manager.start_capture(req.model, req.weights, sample, req.modules, req.max_capture_mb,
                                    req.backend)  # fmt: skip
        return job.to_json()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return _meta(run_id)

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, bool]:
        _meta(run_id)
        manager.delete(run_id)
        return {"deleted": True}

    @app.get("/api/runs/{run_id}/modules")
    def modules(run_id: str) -> list[dict[str, Any]]:
        return module_tree(_ir(run_id))

    @app.get("/api/runs/{run_id}/graph")
    def graph(
        run_id: str,
        depth: int = Query(2, ge=0, le=12),
        method: str | None = None,
        target: int | None = None,
        steps: int = 32,
        expand: str = "",
        collapse: str = "",
        obj_op: str | None = None,
        obj_ch: int | None = Query(None, ge=0),
    ) -> dict[str, Any]:
        ir = _ir(run_id)
        if (obj_op is None) != (obj_ch is None):
            raise HTTPException(400, "obj_op and obj_ch go together")
        objective = {"op": obj_op, "channel": obj_ch} if obj_op is not None else None
        res = manager.attribution(run_id, method, target, steps, objective) if method else {}
        calls = [[c for c in s.split(",") if c] for s in (expand, collapse)]
        return graph_view(ir, depth, res.get("scores"), *calls, merges=res.get("merges"), start=obj_op)

    @app.get("/api/runs/{run_id}/index")
    def index(run_id: str) -> list[dict[str, Any]]:
        return search_index(_ir(run_id))

    @app.get("/api/runs/{run_id}/ops/{op_id}")
    def op(run_id: str, op_id: str) -> dict[str, Any]:
        ir = _ir(run_id)
        for o in ir["ops"]:
            if o["id"] == op_id:
                return o
        raise HTTPException(404, f"Op {op_id} not found")

    @app.get("/api/runs/{run_id}/ops/{op_id}/tensor")
    def op_tensor(
        run_id: str,
        op_id: str,
        channel: int | None = None,
        py: int | None = None,
        px: int | None = None,
        norm: Literal["channel", "global"] = "channel",
        cmap: Literal["seq", "div"] = "seq",
        lo: float | None = None,
        hi: float | None = None,
    ) -> dict[str, Any]:
        arr = _tensor(run_id, op_id)
        pixel = (py, px) if py is not None and px is not None else None
        return tensor_view(arr, channel=channel, pixel=pixel, norm=norm, cmap=cmap, lo=lo, hi=hi)

    @app.get("/api/runs/{run_id}/ops/{op_id}/slice")
    def op_slice(run_id: str, op_id: str, s: str = "") -> dict[str, Any]:
        return slice_table(_tensor(run_id, op_id), s) | {"op": op_id}  # SliceError is a ValueError: 400

    @app.get("/api/runs/{run_id}/ops/{op_id}/slice.npy")
    def op_slice_npy(run_id: str, op_id: str, s: str = "") -> Response:
        arr = _tensor(run_id, op_id)
        try:
            data = slice_npy(arr, s)
        except SliceTooLarge as e:
            raise HTTPException(413, str(e)) from e
        headers = {"Content-Disposition": f'attachment; filename="{safe_name(op_id)}.npy"'}
        return Response(data, media_type="application/octet-stream", headers=headers)

    @app.get("/api/runs/{run_id}/diff")
    def diff(run_id: str, a: str, b: str) -> dict[str, Any]:
        return {"a": a, "b": b, **tensor_diff(_tensor(run_id, a), _tensor(run_id, b))}

    @app.get("/api/runs/{run_id}/input.png")
    def input_png(run_id: str) -> FileResponse:
        _meta(run_id)
        path = store.run_dir(run_id) / PREVIEW
        if not path.is_file():
            raise HTTPException(404, f"Run {run_id} has no input preview")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/runs/{run_id}/attribution")
    def attribution(run_id: str, req: AttributionRequest) -> dict[str, Any]:
        _meta(run_id)
        if req.method not in METHODS:
            raise HTTPException(400, f"method must be one of {sorted(METHODS)}")
        objective = req.objective.model_dump() if req.objective else None
        return manager.attribution(run_id, req.method, req.target, req.steps, objective)

    @app.post("/api/runs/{run_id}/jobs")
    def start_job(run_id: str, req: JobRequest) -> dict[str, Any]:
        _meta(run_id)
        return manager.job(run_id, req.model_dump()).to_json()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return _job(job_id).to_json()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        _job(job_id)
        return manager.jobs.cancel(job_id).to_json()

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str) -> StreamingResponse:
        """Server-sent events: ``data: <job json>`` per change, ``: ping`` every 15 s; ends when terminal."""
        _job(job_id)

        def gen():
            version = -1
            while True:
                try:
                    job, new = manager.jobs.wait(job_id, version, 15.0)
                except KeyError:  # dropped from the runner's history
                    return
                if new == version:
                    yield ": ping\n\n"
                    continue
                version = new
                yield f"data: {_JSON.dump_json(job).decode()}\n\n"  # NaN/Inf -> null, as in JSON replies
                if job["status"] in TERMINAL:
                    return

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, f"No API route /{path}")
            f = (STATIC_DIR / path).resolve()
            if path and f.is_relative_to(STATIC_DIR.resolve()) and f.is_file():
                return FileResponse(f)
            # revalidate the shell so a rebuilt frontend never points at deleted hashed assets
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
