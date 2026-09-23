"""HTTP API. Everything the frontend needs, JSON only; tensors leave the server as small summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter

from .attribution import METHODS
from .config import get_settings
from .graph import graph_view, module_tree, search_index
from .jobs import TERMINAL, Job
from .models import SUGGESTED_MODELS
from .runs import RunManager
from .slicing import SliceError, SliceTooLarge, slice_npy, slice_table
from .store import PREVIEW, RunStore, safe_name
from .tensors import tensor_diff, tensor_view

STATIC_DIR = Path(__file__).parent / "static"
_JSON = TypeAdapter(Any)


class Sample(BaseModel):
    kind: str = Field("random", pattern="^(random|upload)$")
    upload_id: str | None = None
    name: str | None = None
    shape: list[int] | None = None
    seed: int = 0


class CreateRun(BaseModel):
    model: str = "torchvision:resnet18"
    weights: str | None = None
    sample: Sample = Sample()


class Objective(BaseModel):
    op: str
    channel: int = Field(ge=0)


class AttributionRequest(BaseModel):
    method: str = "ig"
    target: int | None = None
    steps: int = Field(32, ge=1, le=512)
    objective: Objective | None = None


class JobRequest(BaseModel):
    kind: Literal["attribution", "layer", "neuron", "agreement", "patch"]
    method: str | None = None  # attribution: gxa|saliency|ig (default ig); layer: layer_ig|layer_cond
    target: int | None = None
    objective: Objective | None = None
    steps: int = Field(32, ge=1, le=512)
    op: str | None = None  # layer / neuron / patch; agreement: optional (units block)
    channel: int | None = Field(None, ge=0)  # neuron
    baseline: Literal["zeros", "mean", "run"] = "zeros"  # patch
    source_run: str | None = None  # patch, baseline == "run"


def create_app(manager: RunManager | None = None) -> FastAPI:
    settings = get_settings()
    if manager is None:
        manager = RunManager(
            RunStore(settings.runs_dir), settings.device, settings.max_capture_bytes, settings.max_live_models
        )
    store = manager.store
    app = FastAPI(title="Neurotome", version="0.1.0")
    app.state.manager = manager
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

    def _attribution(run_id: str, method: str, target: int | None, steps: int, objective: dict | None):
        try:
            return manager.attribution(run_id, method, target, steps, objective)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    def _tensor(run_id: str, op_id: str):
        _meta(run_id)
        try:
            return store.load_tensor(run_id, op_id)
        except KeyError as e:
            raise HTTPException(404, e.args[0] if e.args else str(e)) from e

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        import torch

        return {
            "ok": True,
            "device": manager.device,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "runs_dir": str(store.root),
        }

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {"suggested": SUGGESTED_MODELS, "methods": METHODS}

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

    @app.post("/api/runs")
    def create_run(req: CreateRun) -> dict[str, Any]:
        try:
            return manager.create_from_spec(req.model, req.weights, req.sample.model_dump())
        except (ValueError, KeyError, ImportError, AttributeError) as e:
            raise HTTPException(400, str(e)) from e

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
        res = _attribution(run_id, method, target, steps, objective) if method else {}
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
        arr = _tensor(run_id, op_id)
        try:
            return slice_table(arr, s) | {"op": op_id}
        except SliceError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/runs/{run_id}/ops/{op_id}/slice.npy")
    def op_slice_npy(run_id: str, op_id: str, s: str = "") -> Response:
        arr = _tensor(run_id, op_id)
        try:
            data = slice_npy(arr, s)
        except SliceError as e:
            raise HTTPException(400, str(e)) from e
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
        return _attribution(run_id, req.method, req.target, req.steps, objective)

    @app.post("/api/runs/{run_id}/jobs")
    def start_job(run_id: str, req: JobRequest) -> dict[str, Any]:
        _meta(run_id)
        try:
            return manager.job(run_id, req.model_dump()).to_json()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

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
            f = STATIC_DIR / path
            if path and f.is_file():
                return FileResponse(f)
            # revalidate the shell so a rebuilt frontend never points at deleted hashed assets
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
