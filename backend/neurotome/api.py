"""HTTP API. Everything the frontend needs, JSON only; tensors leave the server as small summaries."""

from __future__ import annotations

import importlib.util
import io
import ipaddress
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from . import __version__
from .attribution import METHODS
from .config import get_settings
from .errors import describe
from .graph import graph_view, module_tree, search_index, unsaved
from .jobs import TERMINAL, Job
from .models import SUGGESTED_MODELS, TEXT_MODELS, list_samples
from .runs import MAX_REF, NotFound, RunManager, _module, input_summary, matrix_view, signal_view
from .slicing import SliceTooLarge, slice_npy, slice_table
from .store import PREVIEW, safe_name
from .tensors import input_preview, tensor_diff, tensor_view

STATIC_DIR = Path(__file__).parent / "static"
_JSON = TypeAdapter(Any)
MAX_UPLOAD = 64 * 1024 * 1024
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def local_host(host: str | None, extra: tuple[str, ...] = ()) -> bool:
    """A Host header naming this machine: localhost, an IP literal, or a name in ``extra``. A page on another
    site that rebinds its DNS name to 127.0.0.1 still sends its own name, so it is refused."""
    name = (urlsplit(f"//{host}").hostname or "") if host else ""
    if not host or name in ("localhost", "testserver") or name.endswith(".localhost") or name in extra:
        return True
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


class SameOrigin:
    """Refuse requests whose Host is not this machine (DNS rebinding: another site's page reading runs or
    starting captures), and state-changing requests that a browser sent from another site (its ``Origin`` is
    neither the dev UI nor this server): a web page must not upload files or start captures (``py:`` specs run
    code)."""

    def __init__(self, app: Any, hosts: tuple[str, ...] = ()):
        self.app = app
        self.hosts = hosts

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            h = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            if not local_host(h.get("host"), self.hosts):
                detail = (f"Host {h.get('host')!r} refused: open Neurotome at http://127.0.0.1, or list "
                          "the name in NEUROTOME_ALLOWED_HOSTS")  # fmt: skip
                await JSONResponse({"detail": detail}, 403)(scope, receive, send)
                return
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            origin = h.get("origin")
            if origin and origin not in DEV_ORIGINS and urlsplit(origin).netloc != h.get("host"):
                await JSONResponse({"detail": "Cross-origin request refused"}, 403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class Sample(BaseModel):
    kind: str = Field(default="random", pattern="^(random|upload|text|ref)$")
    upload_id: str | None = None
    text: str | None = Field(default=None, max_length=10000)  # kind text
    ref: str | None = Field(default=None, max_length=MAX_REF)  # kind ref: a py: module's load_sample(ref)
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


class Direction(BaseModel):
    kind: Literal["contrast", "channel"]
    run_a: str | None = None  # contrast
    run_b: str | None = None
    pool: Literal["mean", "last"] = "mean"
    channel: int | None = Field(None, ge=0)  # channel


class EditIn(BaseModel):
    op: str
    kind: Literal["zero", "mean", "resample", "constant", "scale", "steer"]
    channels: list[int] | None = None  # None = every channel
    positions: Literal["last"] | list[int] | list[tuple[int, int]] | None = None
    over: Literal["positions", "runs"] = "positions"  # mean
    source_run: str | None = None  # resample: a run id or "random"
    seed: int = 0  # resample "random"
    value: float = Field(0.0, allow_inf_nan=False)  # constant
    alpha: float = Field(1.0, allow_inf_nan=False)  # scale, steer
    direction: Direction | None = None  # steer


class InterventionSetIn(BaseModel):
    edits: list[EditIn]
    target: int | None = Field(None, ge=0)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # the M3 patch fields (baseline, source_run) are gone: 422

    kind: Literal["attribution", "layer", "neuron", "agreement", "lens", "intervene", "generate"]
    method: str | None = None  # attribution: gxa|saliency|ig (default ig); layer: layer_ig|layer_cond
    target: int | None = None  # lens: a token id, None = the top prediction
    objective: Objective | None = None
    steps: int = Field(32, ge=1, le=512)
    op: str | None = None  # layer / neuron; agreement: optional (units block)
    channel: int | None = Field(None, ge=0)  # neuron
    interventions: InterventionSetIn | None = None  # intervene, generate
    max_new_tokens: int = Field(16, ge=1, le=64)  # generate


class Rename(BaseModel):
    name: str = Field(max_length=80)


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
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SameOrigin, hosts=(settings.host.lower(), *settings.allowed_hosts))

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

    def _tensor(run_id: str, op_id: str, iv: str | None = None):
        """The op's tensor; with ``iv`` read through that intervention (``RunManager.iv_tensor``)."""
        _meta(run_id)
        try:
            return manager.iv_tensor(run_id, iv, op_id) if iv else store.load_tensor(run_id, op_id)
        except NotFound as e:
            raise HTTPException(404, e.args[0]) from None
        except KeyError:
            raise HTTPException(404, _no_tensor(run_id, op_id)) from None

    def _found(fn, *args):
        try:
            return fn(*args)
        except NotFound as e:
            raise HTTPException(404, e.args[0]) from None

    def _no_tensor(run_id: str, op_id: str) -> str:
        ir = store.ir(run_id)
        op = next((o for o in ir["ops"] if o["id"] == op_id), None)
        if op is None:
            return f"Op {op_id} not found in {run_id}"
        why, model = unsaved(ir, op), ir["model"]
        if why == "budget":
            budget = model.get("budget_bytes")
            budget = (manager.max_capture_bytes if budget is None else budget) or 0
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

    @app.get("/api/samples")
    def samples(model: str) -> dict[str, Any]:
        """The dataset samples a ``py:`` model module lists (``list_samples()``); [] for other specs. A module
        that fails to import or list answers 200 with its error, so the form can say why."""
        try:
            return {"samples": list_samples(model), "error": None}
        except Exception as e:  # noqa: BLE001 - arbitrary module code: its message, never a 500
            return {"samples": [], "error": describe(e)[1]}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return store.list_runs()

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:
        data = await file.read(MAX_UPLOAD + 1)  # never the whole of an oversized file in memory
        if len(data) > MAX_UPLOAD:
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
        steps: int = Query(32, ge=1, le=512),  # as AttributionRequest: this route can compute attribution
        expand: str = "",
        collapse: str = "",
        obj_op: str | None = None,
        obj_ch: int | None = Query(None, ge=0),
        iv: str | None = None,
    ) -> dict[str, Any]:
        """``iv``: ops the intervention recorded carry its stats (colour modes, the inspector histogram)."""
        _meta(run_id)
        ir = _found(manager.iv_ir, run_id, iv) if iv else store.ir(run_id)
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
        iv: str | None = None,
    ) -> dict[str, Any]:
        arr = _tensor(run_id, op_id, iv)
        pixel = (py, px) if py is not None and px is not None else None
        return tensor_view(arr, channel=channel, pixel=pixel, norm=norm, cmap=cmap, lo=lo, hi=hi)

    @app.get("/api/runs/{run_id}/ops/{op_id}/slice")
    def op_slice(run_id: str, op_id: str, s: str = "", iv: str | None = None) -> dict[str, Any]:
        return slice_table(_tensor(run_id, op_id, iv), s) | {"op": op_id}  # SliceError is a ValueError: 400

    @app.get("/api/runs/{run_id}/ops/{op_id}/slice.npy")
    def op_slice_npy(run_id: str, op_id: str, s: str = "", iv: str | None = None) -> Response:
        arr = _tensor(run_id, op_id, iv)
        try:
            data = slice_npy(arr, s)
        except SliceTooLarge as e:
            raise HTTPException(413, str(e)) from e
        headers = {"Content-Disposition": f'attachment; filename="{safe_name(op_id)}.npy"'}
        return Response(data, media_type="application/octet-stream", headers=headers)

    @app.get("/api/runs/{run_id}/diff")
    def diff(run_id: str, a: str, b: str, a_iv: str | None = None, b_iv: str | None = None) -> dict[str, Any]:
        return {"a": a, "b": b, **tensor_diff(_tensor(run_id, a, a_iv), _tensor(run_id, b, b_iv))}

    @app.get("/api/runs/{run_id}/input.png")
    def input_png(run_id: str) -> FileResponse:
        _meta(run_id)
        path = store.run_dir(run_id) / PREVIEW
        if not path.is_file():
            raise HTTPException(404, f"Run {run_id} has no input preview")
        return FileResponse(path, media_type="image/png")

    def _input_op(run_id: str) -> dict[str, Any]:
        op = next((o for o in store.ir(run_id)["ops"] if o["is_input"]), None)
        if op is None:
            raise HTTPException(404, f"Run {run_id} has no input op")
        return op

    @app.get("/api/runs/{run_id}/input")
    def input_info(run_id: str) -> dict[str, Any]:
        """The Input view: the raw sample, the preprocessing in words, the model input's shape and ranges."""
        meta, op = _meta(run_id), _input_op(run_id)
        x = np.asarray(_tensor(run_id, op["id"]))
        info = meta.get("input") or {}  # absent before 0.3.0: no raw copy, no steps
        pic = x.ndim == 4 and x.shape[1] in (1, 3) and min(x.shape[2:]) >= 8
        return {"op": op["id"], "shape": op["shape"], "dtype": op.get("dtype"), "sample": meta["sample"],
                "raw": info.get("raw"), "steps": info.get("steps"),
                "normalised": info.get("norm") is not None,
                "picture": bool(pic), "channels": input_summary(x, info.get("blocks")),
                "blocks": info.get("blocks"), "time": info.get("time"),
                "info": meta["sample"].get("info")}  # fmt: skip

    @app.get("/api/runs/{run_id}/input/signal")
    def input_signal(run_id: str, points: int = Query(1500, ge=16, le=20000)) -> dict[str, Any]:
        """The raw signal of a dataset sample (``load_sample`` raw): min / max envelope + mean per channel."""
        raw = (_meta(run_id).get("input") or {}).get("raw")
        if not raw or raw.get("kind") != "signal":
            raise HTTPException(404, f"Run {run_id} kept no raw signal")
        return signal_view(store.run_dir(run_id), raw, points)

    @app.get("/api/runs/{run_id}/input/matrix")
    def input_matrix(run_id: str) -> dict[str, Any]:
        """A float (1, T, C) model input as one time × column heatmap per block (``load_sample`` blocks)."""
        info = _meta(run_id).get("input") or {}
        x = np.asarray(_tensor(run_id, _input_op(run_id)["id"]))
        if x.ndim != 3 or x.shape[0] != 1 or not np.issubdtype(x.dtype, np.floating):
            raise HTTPException(404, f"The input of {run_id} is not a float (1, T, C) tensor")
        return matrix_view(x, info.get("blocks"), info.get("time"))

    @app.get("/api/runs/{run_id}/input/raw")
    def input_raw(run_id: str) -> FileResponse:
        raw = (_meta(run_id).get("input") or {}).get("raw")
        path = store.run_dir(run_id) / Path(raw["file"]).name if raw else None
        if path is None or not path.is_file():
            raise HTTPException(404, f"Run {run_id} kept no raw input (uploaded images, dataset signals)")
        mime = {".jpg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
                ".npz": "application/octet-stream"}  # fmt: skip
        return FileResponse(path, media_type=mime.get(path.suffix, "image/png"))  # not the OS's guess

    @app.get("/api/runs/{run_id}/input/model.png")
    def input_model_png(run_id: str) -> Response:
        """The model input at its own resolution (<= 1024 px), de-normalised for display."""
        norm = (_meta(run_id).get("input") or {}).get("norm")
        x = np.asarray(_tensor(run_id, _input_op(run_id)["id"]), dtype=np.float32)
        img = input_preview(x, (norm[0], norm[1]) if norm else None)
        if img is None:
            raise HTTPException(404, f"The input of {run_id} is not an image")
        from PIL import Image

        pic = Image.fromarray(img)
        pic.thumbnail((1024, 1024))
        buf = io.BytesIO()
        pic.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/api/runs/{run_id}/output")
    def output(run_id: str) -> dict[str, Any]:
        """Every output element with its label (``RunManager.output``): the Trace view's step picker."""
        _meta(run_id)
        return manager.output(run_id)

    @app.get("/api/runs/{run_id}/windows")
    def windows(run_id: str, t: int = Query(..., ge=0)) -> dict[str, Any]:
        """The input windows of output element ``t``: the py: module's ``target_windows(t)``, checked and
        clipped (``runs.check_windows``); [] without the hook; a failing or malformed hook answers 200 with
        its error."""
        _meta(run_id)
        return manager.windows(run_id, t)

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

    @app.get("/api/runs/{run_id}/interventions")
    def interventions(run_id: str) -> list[dict[str, Any]]:
        _meta(run_id)
        return manager.interventions(run_id)

    @app.get("/api/runs/{run_id}/interventions/{h}")
    def intervention(run_id: str, h: str) -> dict[str, Any]:
        _meta(run_id)
        return _found(manager.intervention, run_id, h)

    @app.patch("/api/runs/{run_id}/interventions/{h}")
    def rename_intervention(run_id: str, h: str, req: Rename) -> dict[str, Any]:
        _meta(run_id)
        return _found(manager.rename, run_id, h, req.name)

    @app.delete("/api/runs/{run_id}/interventions/{h}")
    def delete_intervention(run_id: str, h: str) -> dict[str, bool]:
        _meta(run_id)
        _found(manager.delete_intervention, run_id, h)
        return {"deleted": True}

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
            # an anchored path (C:\, \\host\share) is never joined: resolving a UNC path on Windows
            # would contact that host (and offer it the user's NTLM credentials)
            f = None if Path(path).anchor else (STATIC_DIR / path).resolve()
            if path and f and f.is_relative_to(STATIC_DIR.resolve()) and f.is_file():
                return FileResponse(f)
            # revalidate the shell so a rebuilt frontend never points at deleted hashed assets
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
