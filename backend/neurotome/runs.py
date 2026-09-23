"""Run manager: creates runs (load model -> build input -> capture) and keeps live models for attribution."""

from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .attribution import (
    LAYER_METHODS,
    METHODS,
    NEURON_LABEL,
    agreement,
    attribute,
    channel_conductance,
    input_map,
    layer_attribution,
    logit_label,
    objective_forward,
    unit_label,
)
from .capture import capture, patch
from .errors import ModelUnavailable, describe
from .graph import module_calls, module_site, module_sites
from .jobs import Cancelled, Job, JobRunner, Progress
from .models import LoadedModel, check_spec, load_model, to_logits
from .store import PREVIEW, RunStore, safe_name
from .tensors import input_preview


@dataclass
class Live:
    loaded: LoadedModel
    inputs: Any


class RunManager:
    def __init__(
        self,
        store: RunStore,
        device: str = "cpu",
        max_capture_bytes: int | None = None,
        max_live_models: int = 2,
    ):
        self.store = store
        self.device = device
        self.max_capture_bytes = max_capture_bytes
        self.max_live_models = max_live_models
        self._live: OrderedDict[str, Live] = OrderedDict()  # LRU: most recently used last
        self._lock = threading.Lock()  # one capture / attribution at a time (GPU memory)
        self.uploads = store.root / "_uploads"
        self.uploads.mkdir(exist_ok=True)
        self.jobs = JobRunner(describe=lambda e: describe(e)[1])

    # ---- inputs -----------------------------------------------------------
    def save_upload(self, filename: str, data: bytes) -> str:
        import hashlib

        suffix = Path(filename).suffix.lower()[:8] or ".bin"
        uid = hashlib.sha1(data).hexdigest()[:12] + suffix
        (self.uploads / uid).write_bytes(data)
        return uid

    def build_input(self, loaded: LoadedModel, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        kind = sample.get("kind", "random")
        if kind == "upload":
            uid = Path(sample["upload_id"]).name
            path = self.uploads / uid
            if path.suffix in (".pt", ".pth"):
                x = torch.load(path, map_location="cpu")
                return x, {"kind": "tensor", "name": sample.get("name") or uid}
            if path.suffix == ".npy":
                a = np.load(path)
                integer = np.issubdtype(a.dtype, np.integer)  # token ids stay int64
                x = torch.from_numpy(a.astype(np.int64)) if integer else torch.from_numpy(a).float()
                return x, {"kind": "tensor", "name": sample.get("name") or uid}
            from PIL import Image

            if loaded.preprocess is None:
                raise ValueError(
                    "This model has no image preprocessing; upload a .pt or .npy tensor instead."
                )
            img = Image.open(io.BytesIO(path.read_bytes()))
            return loaded.preprocess(img), {"kind": "image", "name": sample.get("name") or uid}
        shape = sample.get("shape") or loaded.default_input_shape
        g = torch.Generator().manual_seed(int(sample.get("seed", 0)))
        return torch.randn(*shape, generator=g), {"kind": "random", "name": f"randn{tuple(shape)}"}

    # ---- runs -------------------------------------------------------------
    def create_from_spec(
        self,
        spec: str,
        weights: str | None,
        sample: dict[str, Any],
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """``progress`` None is the synchronous path (CLI); a capture job passes its progress callback."""
        report = progress or _quiet
        report(0.02, "Load model")
        try:
            loaded = load_model(spec, weights, self.device)
        except (ImportError, AttributeError, KeyError, TypeError) as e:
            raise ValueError(f"Cannot load model {spec!r}: {type(e).__name__}: {e}") from e
        report(0.08, "Build input")
        x, sample_meta = self.build_input(loaded, sample)
        return self._create(loaded, x, sample_meta, modules, max_capture_mb, progress)

    def create_from_objects(
        self,
        model: torch.nn.Module,
        inputs: Any,
        name: str | None = None,
        categories: list[str] | None = None,
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        loaded = LoadedModel(model=model.eval(), spec=None, weights=None, categories=categories)
        sample_meta = {"kind": "tensor", "name": name or "in-process input"}
        return self._create(loaded, inputs, sample_meta, modules, max_capture_mb, progress)

    def start_capture(
        self,
        spec: str,
        weights: str | None,
        sample: dict[str, Any],
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
    ) -> Job:
        """Queue a capture job (its result is the run meta) after the checks that need no model."""
        check_spec(spec)
        uid = Path(sample.get("upload_id") or "").name
        if sample.get("kind") == "upload" and not (uid and (self.uploads / uid).is_file()):
            raise ValueError(f"Upload {sample.get('upload_id')} not found; upload the file again")
        return self.jobs.submit(
            "capture", lambda p: self.create_from_spec(spec, weights, sample, modules, max_capture_mb, p)
        )

    def _create(
        self,
        loaded: LoadedModel,
        inputs: Any,
        sample_meta: dict[str, Any],
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        modules = [m.strip() for m in modules or [] if m.strip()] or None
        if max_capture_mb is not None and not (math.isfinite(max_capture_mb) and max_capture_mb > 0):
            raise ValueError(f"max_capture_mb must be a positive finite number, got {max_capture_mb}")
        budget = int(max_capture_mb * 1024 * 1024) if max_capture_mb else self.max_capture_bytes
        with self._lock:
            run_id = self.store.new_run_id()
            run_dir = self.store.create(run_id)
            try:
                inputs = _to_device(inputs, self.device)
                ir = capture(loaded.model, inputs, self.store, run_id, budget, modules, progress)
                (progress or _quiet)(0.93, "Summarise")
                cpu = _to_device(inputs, "cpu")
                torch.save(cpu, run_dir / "inputs.pt")
                preview = _save_preview(cpu, loaded.norm if sample_meta["kind"] == "image" else None, run_dir)
                top, output_error = _top_k(loaded, inputs)
            except Exception as e:  # Cancelled and CUDA OOM included: no half-written run is left
                self.store.delete(run_id)
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    _free_cuda()
                raise
            meta = {
                "id": run_id,
                "created": self.store.now(),
                "model": {"spec": loaded.spec, "weights": loaded.weights, **ir["model"]},
                "sample": sample_meta,
                "input_shape": _shape(inputs),
                "output": top,
                "output_error": output_error,
                "preview": preview,
                "categories": loaded.categories,
                "reloadable": loaded.spec is not None,
                "status": "ready",
            }
            self.store.write_json(run_id, "ir.json", ir)
            self.store.write_json(run_id, "meta.json", meta)
            self._keep_live(run_id, Live(loaded, inputs))
            return meta

    def delete(self, run_id: str) -> None:
        with self._lock:
            self.store.delete(run_id)
            if self._live.pop(run_id, None) is not None:
                _free_cuda()

    def _keep_live(self, run_id: str, live: Live) -> Live:
        self._live[run_id] = live
        self._live.move_to_end(run_id)
        evicted = False
        while len(self._live) > max(self.max_live_models, 0):
            self._live.popitem(last=False)
            evicted = True
        if evicted:
            _free_cuda()
        return live

    def live(self, run_id: str) -> Live:
        if run_id in self._live:
            self._live.move_to_end(run_id)
            return self._live[run_id]
        meta = self.store.meta(run_id)
        if not meta.get("reloadable"):
            raise ModelUnavailable(
                "This run was created in-process (neurotome.open) and its model is no longer loaded. Run "
                "neurotome.open(model, x) again to attribute it; the graph and tensors stay browsable."
            )
        spec = meta["model"]["spec"]
        try:
            loaded = load_model(spec, meta["model"]["weights"], self.device)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001 - a module that moved, a renamed factory, ...
            raise ModelUnavailable(
                f"The model of {run_id} ({spec}) can no longer be loaded: {type(e).__name__}: {e}. py: specs "
                "import from the server's working directory; start neurotome from the folder that holds the "
                "module, or capture a new run."
            ) from e
        inputs = torch.load(self.store.run_dir(run_id) / "inputs.pt", map_location=self.device)
        return self._keep_live(run_id, Live(loaded, inputs))

    # ---- attribution: every kind below also runs as a job (see job()) -----
    def _cached(self, run_id: str, name: str) -> Any:
        """The cached result, or None when missing or stale (attribution files before M3 have no merges)."""
        try:
            data = self.store.read_json(run_id, name)
        except KeyError:
            return None
        stale = name.split("/")[-1].split("_")[0] in METHODS and "merges" not in data
        return None if stale else data

    @staticmethod
    def _site(ir: dict[str, Any], op: str, why: str) -> tuple[str, int]:
        if not any(o["id"] == op for o in ir["ops"]):
            raise ValueError(f"Unknown op {op}")
        site = module_site(ir, op)
        if site is None:
            raise ValueError(f"{op} is not the output of a module; pick its module node ({why})")
        return site

    def _layer_site(self, ir: dict[str, Any], op: str) -> tuple[str, int]:
        """The innermost module called once whose output is ``op`` (a block's last ReLU is often shared)."""
        site = self._site(ir, op, "layer methods hook module outputs")
        once = [s for s in module_sites(ir, op) if module_calls(ir, s[0]) == 1]
        if not once:
            n = module_calls(ir, site[0])
            raise ValueError(f"{site[0]} is called {n} times; layer methods need a module called once")
        return once[0]

    def _objective(self, live: Live, ir: dict[str, Any], target: int | None, objective: dict | None,
                   layer_op: str | None = None):  # fmt: skip
        """(fwd, target, label) for Captum: a logit (argmax when target is None) or a unit objective.
        ``layer_op``: the layer being attributed, which must lie strictly before a unit objective."""
        model = live.loaded.model
        if objective:
            site = self._site(ir, objective["op"], UNIT_WHY)
            if layer_op is not None and layer_op not in _ancestors(ir, objective["op"]):
                raise ValueError(
                    f"{layer_op} does not feed the objective {objective['op']}; pick a layer before it"
                )
            fwd, t = objective_forward(model, site, int(objective["channel"]))
            return fwd, t, unit_label(site[0], int(objective["channel"]))
        if target is None:
            with torch.no_grad():
                target = int(to_logits(model(_first(live.inputs)))[0].argmax())
        fwd, _ = objective_forward(model, None, None)
        return fwd, target, logit_label(target, live.loaded.categories)

    def attribution(
        self,
        run_id: str,
        method: str,
        target: int | None,
        steps: int,
        objective: dict[str, Any] | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        name = cache_name("attribution", method, target, objective, steps)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        with self._lock:
            live = self.live(run_id)
            ir = self.store.ir(run_id)
            site = self._site(ir, objective["op"], UNIT_WHY) if objective else None
            unit_ops = {o["id"] for o in ir["ops"] if o["num_params"] > 0 and o["type"] not in ("batchnorm",)}
            result = attribute(
                live.loaded.model,
                live.inputs,
                method,
                target,
                steps,
                live.loaded.categories,
                unit_ops or None,
                objective,
                site,
                progress,
            )
        where = {o["id"]: o.get("module") or o["id"] for o in ir["ops"]}
        for u in result["top_units"]:
            u["path"] = where.get(u["op"], u["op"])
        if site and objective:
            result["objective_label"] = unit_label(site[0], int(objective["channel"]))
        channels = result.pop("channel_scores")
        self.store.write_json(run_id, f"attribution/channels_{okey(target, objective)}.json", channels)
        self.store.write_json(run_id, name, result)
        return result

    def layer(
        self,
        run_id: str,
        method: str,
        op: str,
        target: int | None,
        steps: int,
        objective: dict[str, Any] | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        if method not in LAYER_METHODS:
            raise ValueError(f"method must be one of {sorted(LAYER_METHODS)}")
        name = cache_name("layer", method, target, objective, steps, op)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        with self._lock:
            live = _float_input(self.live(run_id))
            ir = self.store.ir(run_id)
            address, _ = self._layer_site(ir, op)
            fwd, t, label = self._objective(live, ir, target, objective, op)
            model = live.loaded.model
            layer = model.get_submodule(address)
            res = layer_attribution(model, _first(live.inputs), layer, method, fwd, t, steps, progress)
        result = {
            "method": method,
            "method_label": LAYER_METHODS[method],
            "op": op,
            "path": address,
            "objective_label": label,
            "steps": steps,
            **res,
        }
        self.store.write_json(run_id, name, result)
        return result

    def neuron(
        self,
        run_id: str,
        op: str,
        channel: int,
        target: int | None,
        steps: int,
        objective: dict[str, Any] | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        name = cache_name("neuron", None, target, objective, steps, op, channel)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        with self._lock:
            live = _float_input(self.live(run_id))
            ir = self.store.ir(run_id)
            address, _ = self._layer_site(ir, op)
            shape = next(o["shape"] for o in ir["ops"] if o["id"] == op)
            if len(shape) < 2 or not 0 <= channel < shape[1]:
                raise ValueError(f"Channel {channel} out of range for {op} with shape {shape}")
            fwd, t, label = self._objective(live, ir, target, objective, op)
            model = live.loaded.model
            layer = model.get_submodule(address)
            t = cast(int, t)  # a logit index or 0 (unit objective), never None here
            attr = channel_conductance(model, _first(live.inputs), layer, channel, fwd, t, steps, progress)
        result = {
            "method": "neuron_cond",
            "method_label": NEURON_LABEL,
            "op": op,
            "path": address,
            "channel": channel,
            "objective_label": label,
            "steps": steps,
            "input": input_map(attr, None),
        }
        self.store.write_json(run_id, name, result)
        return result

    def agreement(
        self,
        run_id: str,
        target: int | None,
        steps: int,
        objective: dict[str, Any] | None = None,
        op: str | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """The input methods (and, with ``op``, the per-channel methods) compared. Never holds _lock itself:
        it runs or reuses the other methods, which take it."""
        name = cache_name("agreement", None, target, objective, steps, op)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        report = progress or _quiet
        total = len(METHODS) + (len(LAYER_METHODS) if op else 0)

        def part(i: int, m: str) -> Progress:
            msg = f"Agreement · {m} ({i + 1}/{total})"
            report(i / total, msg)
            return lambda f, _msg: report((i + f) / total, msg)

        res = {m: self.attribution(run_id, m, target, steps, objective, part(i, m))
               for i, m in enumerate(METHODS)}  # fmt: skip
        maps = {m: np.asarray(r["input"]["map"]) for m, r in res.items() if r["input"]}
        block = agreement(maps, METHODS, "|attribution| · 24×24 pooled input map") if len(maps) > 1 else None
        units, errors = None, []
        if op:
            vecs: dict[str, Any] = {}
            channels = self.store.read_json(run_id, f"attribution/channels_{okey(target, objective)}.json")
            if op in channels:
                vecs["gxa"] = channels[op]
            else:
                errors.append(f"no per-channel gradient x activation for {op}")
            for i, m in enumerate(LAYER_METHODS, start=len(METHODS)):
                try:
                    vecs[m] = self.layer(run_id, m, op, target, steps, objective, part(i, m))["channels"]
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - a failing method is left out and named
                    if str(e) not in errors:
                        errors.append(str(e))
            if len(vecs) >= 2:
                site = module_site(self.store.ir(run_id), op)
                labels = {"gxa": "Gradient x activation", **LAYER_METHODS}
                units = agreement(vecs, labels, f"|attribution| · per channel of {site[0] if site else op}")
                units["op"] = op
        result = {
            "objective_label": res["ig"]["objective_label"],
            "steps": steps,
            "input": block,
            "units": units,
            "units_error": "; ".join(errors) or None,
        }
        self.store.write_json(run_id, name, result)
        return result

    def patch(
        self,
        run_id: str,
        op: str,
        baseline: str,
        source_run: str | None = None,
        target: int | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Replace one op's output (zeros, its spatial mean, or the same op of another run of this model)
        and replay the forward pass."""
        name = cache_name("patch", None, target, None, 0, op, baseline=baseline, source_run=source_run)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        report = progress or _quiet
        if baseline == "run":
            model, src_id = self.store.meta(run_id)["model"], source_run or ""
            other = self.store.meta(src_id)["model"] if self.store.exists(src_id) else {}
            same = [(m.get("spec"), m.get("weights")) for m in (model, other)]
            if source_run == run_id or model.get("spec") is None or same[0] != same[1]:
                raise ValueError("source run must be a different run with the same model spec and weights")
            try:
                src = np.array(self.store.load_tensor(src_id, op))
            except KeyError as e:
                raise ValueError(f"source run has no saved tensor for {op}") from e
            replace = lambda t: torch.as_tensor(src, dtype=t.dtype, device=t.device)  # noqa: E731
        elif baseline == "mean":
            replace = _spatial_mean
        elif baseline == "zeros":
            replace = torch.zeros_like
        else:
            raise ValueError("baseline must be zeros, mean or run")
        with self._lock:
            live = self.live(run_id)
            ir = self.store.ir(run_id)
            if not any(o["id"] == op for o in ir["ops"]):
                raise ValueError(f"Unknown op {op}")
            model = live.loaded.model.eval()
            report(0.05, "Clean pass")
            with torch.no_grad():
                args = live.inputs if isinstance(live.inputs, (tuple, list)) else (live.inputs,)
                logits = to_logits(model(*args))[0].float()
            t = int(logits.argmax()) if target is None else target
            report(0.3, "Patched replay")
            out = patch(model, live.inputs, op, replace)
            after = to_logits(out["logits_after"])[0].float()
        cats = live.loaded.categories
        p0, p1 = torch.softmax(logits, 0), torch.softmax(after, 0)
        before_ = {"logit": float(logits[t]), "prob": float(p0[t])}
        after_ = {"logit": float(after[t]), "prob": float(p1[t])}
        site = module_site(ir, op)
        result = {
            "op": op,
            "path": site[0] if site else op,
            "baseline": baseline,
            "source_run": source_run if baseline == "run" else None,
            "method_label": "Activation patching · TorchLens replay",
            "target": t,
            "target_label": cats[t] if cats and t < len(cats) else None,
            "before": before_,
            "after": after_,
            "delta_logit": after_["logit"] - before_["logit"],
            "delta_prob": after_["prob"] - before_["prob"],
            "top_after": _top_of(after, cats),
            "deltas": out["deltas"],
            "rel": out["rel"],
        }
        self.store.write_json(run_id, name, result)
        return result

    def job(self, run_id: str, req: dict[str, Any]) -> Job:
        """Start one job, or answer it at once from the cache. ValueError for a bad request (HTTP 400)."""
        self.store.meta(run_id)
        kind, target, steps = req.get("kind"), req.get("target"), int(req.get("steps") or 32)
        obj, op, channel = req.get("objective") or None, req.get("op"), req.get("channel")
        if kind == "attribution":
            method = req.get("method") or "ig"
            if method not in METHODS:
                raise ValueError(f"method must be one of {sorted(METHODS)}")
            name = cache_name(kind, method, target, obj, steps)
            fn = lambda p: self.attribution(run_id, method, target, steps, obj, p)  # noqa: E731
        elif kind == "layer":
            method = req.get("method") or "layer_ig"
            if not op or method not in LAYER_METHODS:
                raise ValueError(f"layer jobs need op and a method in {sorted(LAYER_METHODS)}")
            name = cache_name(kind, method, target, obj, steps, op)
            fn = lambda p: self.layer(run_id, method, op, target, steps, obj, p)  # noqa: E731
        elif kind == "neuron":
            if not op or channel is None:
                raise ValueError("neuron jobs need op and channel")
            name = cache_name(kind, None, target, obj, steps, op, channel)
            fn = lambda p: self.neuron(run_id, op, int(channel), target, steps, obj, p)  # noqa: E731
        elif kind == "agreement":
            name = cache_name(kind, None, target, obj, steps, op)
            fn = lambda p: self.agreement(run_id, target, steps, obj, op, p)  # noqa: E731
        elif kind == "patch":
            baseline, source = req.get("baseline") or "zeros", req.get("source_run")
            if not op or (baseline == "run" and not source):
                raise ValueError("patch jobs need op, and source_run when baseline is run")
            name = cache_name(kind, None, target, None, 0, op, baseline=baseline, source_run=source)
            fn = lambda p: self.patch(run_id, op, baseline, source, target, p)  # noqa: E731
        else:
            raise ValueError(f"Unknown job kind {kind!r}")
        if (cached := self._cached(run_id, name)) is not None:
            return self.jobs.finished(kind, cached, run_id)
        return self.jobs.submit(kind, fn, run_id, key=name)


UNIT_WHY = "unit objectives hook module outputs"


def _quiet(_frac: float, _msg: str) -> None:
    pass


def okey(target: int | None, objective: dict[str, Any] | None) -> str:
    """Cache key of an objective: "top", "<target>" or "u_<op>_c<channel>"."""
    if objective:
        return f"u_{safe_name(objective['op'])}_c{int(objective['channel'])}"
    return "top" if target is None else str(int(target))


def cache_name(
    kind: str,
    method: str | None,
    target: int | None,
    objective: dict[str, Any] | None,
    steps: int,
    op: str | None = None,
    channel: int | None = None,
    baseline: str | None = None,
    source_run: str | None = None,
) -> str:
    """The run-folder file that caches one job's result; also the job's dedup key."""
    k, s = okey(target, objective), safe_name(op or "")
    if kind == "attribution":
        return f"attribution/{method}_{k}_{steps}.json"
    if kind == "layer":
        return f"attribution/{method}_{k}_{s}_{steps}.json"
    if kind == "neuron":
        return f"attribution/neuron_{k}_{s}_c{channel}_{steps}.json"
    if kind == "agreement":
        return f"attribution/agreement_{k}_{steps}{'_' + s if op else ''}.json"
    src = f"_{safe_name(source_run or '')}" if baseline == "run" else ""
    return f"patch/{s}_{baseline}{src}_{'top' if target is None else target}.json"


def _first(inputs: Any) -> Any:
    return inputs[0] if isinstance(inputs, (tuple, list)) else inputs


def _float_input(live: Live) -> Live:
    """``live`` when its input is a float tensor: layer methods interpolate from a zero baseline."""
    x = _first(live.inputs)
    if not (torch.is_tensor(x) and x.is_floating_point()):
        raise ValueError(
            "Layer methods interpolate the input from a zero baseline and need a float input; this run's "
            f"input is {getattr(x, 'dtype', type(x).__name__)}. Embedding-layer methods are planned for v0.2."
        )
    return live


def _spatial_mean(t: torch.Tensor) -> torch.Tensor:
    """Mean over dims >= 2 (rank >= 3) or over the last dim (rank <= 2), broadcast back to t's shape."""
    dims = tuple(range(2, t.dim())) or (t.dim() - 1,)
    return t.mean(dim=dims, keepdim=True).expand_as(t).clone()


def _free_cuda() -> None:
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _to_device(inputs: Any, device: str) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, (tuple, list)):
        return type(inputs)(_to_device(i, device) for i in inputs)
    return inputs


def _save_preview(inputs: Any, norm: Any, run_dir: Path) -> dict[str, int] | None:
    """Write input_preview.png (<= 224 px) for image-like inputs; returns its size or None."""
    try:  # optional artifact: a failure only disables the overlay, never the capture
        x = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
        img = input_preview(x.detach().float().numpy(), norm) if torch.is_tensor(x) else None
        if img is None:
            return None
        from PIL import Image

        pic = Image.fromarray(img)
        pic.thumbnail((224, 224))
        pic.save(run_dir / PREVIEW)
        return {"width": pic.width, "height": pic.height}
    except Exception:  # noqa: BLE001 - sparse/quantized inputs, PIL errors
        (run_dir / PREVIEW).unlink(missing_ok=True)
        return None


def _shape(inputs: Any) -> list[int] | None:
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    return list(x.shape) if torch.is_tensor(x) else None


def _top_k(loaded: LoadedModel, inputs: Any, k: int = 5) -> tuple[list[dict[str, Any]], str | None]:
    """(top-k, None), or ([], why) when the output gives no logits: the run is kept either way."""
    try:
        with torch.no_grad():
            args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
            logits = to_logits(loaded.model(*args))[0].float()
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:  # noqa: BLE001 - exotic outputs: skip the summary, keep the run
        return [], describe(e)[1]
    return _top_of(logits, loaded.categories, k), None


def _top_of(logits: torch.Tensor, cats: list[str] | None, k: int = 5) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=0)
    vals, idx = probs.topk(min(k, probs.numel()))
    return [
        {
            "index": int(i),
            "prob": float(v),
            "logit": float(logits[i]),
            "label": cats[int(i)] if cats and int(i) < len(cats) else None,
        }
        for v, i in zip(vals, idx, strict=True)
    ]


def _ancestors(ir: dict[str, Any], op: str) -> set[str]:
    """Every op that feeds ``op`` (not ``op`` itself)."""
    parents = {o["id"]: o["parents"] for o in ir["ops"]}
    seen: set[str] = set()
    todo = list(parents.get(op, []))
    while todo:
        if (p := todo.pop()) not in seen:
            seen.add(p)
            todo.extend(parents.get(p, []))
    return seen
