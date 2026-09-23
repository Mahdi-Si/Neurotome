"""HTTP API. Everything the frontend needs, JSON only; tensors leave the server as small summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .attribution import METHODS
from .config import get_settings
from .graph import graph_view, module_tree
from .models import SUGGESTED_MODELS
from .runs import RunManager
from .store import RunStore
from .tensors import tensor_view

STATIC_DIR = Path(__file__).parent / "static"


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


class AttributionRequest(BaseModel):
    method: str = "ig"
    target: int | None = None
    steps: int = Field(32, ge=1, le=512)


def create_app(manager: RunManager | None = None) -> FastAPI:
    settings = get_settings()
    if manager is None:
        manager = RunManager(RunStore(settings.runs_dir), settings.device, settings.max_capture_bytes)
    store = manager.store
    app = FastAPI(title="Dissect", version="0.1.0")
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
        store.delete(run_id)
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
    ) -> dict[str, Any]:
        ir = _ir(run_id)
        scores = None
        if method:
            scores = manager.attribution(run_id, method, target, steps)["scores"]
        return graph_view(ir, depth, scores)

    @app.get("/api/runs/{run_id}/ops/{op_id}")
    def op(run_id: str, op_id: str) -> dict[str, Any]:
        ir = _ir(run_id)
        for o in ir["ops"]:
            if o["id"] == op_id:
                return o
        raise HTTPException(404, f"Op {op_id} not found")

    @app.get("/api/runs/{run_id}/ops/{op_id}/tensor")
    def op_tensor(
        run_id: str, op_id: str, channel: int | None = None, py: int | None = None, px: int | None = None
    ) -> dict[str, Any]:
        _meta(run_id)
        try:
            arr = store.load_tensor(run_id, op_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        pixel = (py, px) if py is not None and px is not None else None
        return tensor_view(arr, channel=channel, pixel=pixel)

    @app.post("/api/runs/{run_id}/attribution")
    def attribution(run_id: str, req: AttributionRequest) -> dict[str, Any]:
        _meta(run_id)
        if req.method not in METHODS:
            raise HTTPException(400, f"method must be one of {sorted(METHODS)}")
        try:
            return manager.attribution(run_id, req.method, req.target, req.steps)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            f = STATIC_DIR / path
            return FileResponse(f if path and f.is_file() else STATIC_DIR / "index.html")

    return app
