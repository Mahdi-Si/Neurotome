"""Run manager: creates runs (load model -> build input -> capture) and keeps live models for attribution."""

from __future__ import annotations

import importlib
import io
import json
import math
import random
import shutil
import threading
from collections import OrderedDict
from collections.abc import Sequence
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
from .capture import capture
from .errors import ModelUnavailable, describe
from .graph import module_calls, module_site, module_sites
from .interventions import (
    CH_AXIS,
    GEN_KINDS,
    MAX_NEW_TOKENS,
    downstream,
    edit_fn,
    keep_ops,
    label,
    normalise,
    set_hash,
)
from .interventions import generate as generate_tokens
from .jobs import Cancelled, Job, JobRunner, Progress
from .lens import logit_lens
from .models import (
    LoadedModel,
    Vocab,
    _prepare_hf,
    check_spec,
    is_hf,
    is_lm,
    load_model,
    to_logits,
    vocab_size,
)
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
        nnsight_params: int = 1_000_000_000,
    ):
        self.store = store
        self.device = device
        self.max_capture_bytes = max_capture_bytes
        self.max_live_models = max_live_models
        self.nnsight_params = nnsight_params  # auto backend: NNsight above this many parameters
        self._live: OrderedDict[str, Live] = OrderedDict()  # LRU: most recently used last
        self._lock = threading.Lock()  # one capture / attribution at a time (GPU memory)
        self.uploads = store.root / "_uploads"
        self.uploads.mkdir(exist_ok=True)
        self.jobs = JobRunner(describe=lambda e: describe(e)[1])

    @classmethod
    def from_settings(cls, st: Any) -> RunManager:
        """A manager configured by ``config.Settings`` (server, CLI, ``neurotome.open``)."""
        store = RunStore(st.runs_dir)
        return cls(store, st.device, st.max_capture_bytes, st.max_live_models, st.nnsight_params)

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
        if kind == "text":
            return _text_input(loaded, sample.get("text") or "")
        shape = sample.get("shape") or loaded.default_input_shape
        g = torch.Generator().manual_seed(int(sample.get("seed", 0)))
        n = getattr(getattr(loaded.model, "config", None), "vocab_size", 0)
        n = len(loaded.categories) if isinstance(loaded.categories, Vocab) else n
        if loaded.tokenizer is not None and n:
            x = torch.randint(0, n, tuple(shape), generator=g)
            return x, {"kind": "random", "name": f"random tokens ({shape[-1]})"}
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
        backend: str | None = None,
    ) -> dict[str, Any]:
        """``progress`` None is the synchronous path (CLI); a capture job passes its progress callback."""
        report = progress or _quiet
        report(0.02, "Load model")
        try:
            loaded = load_model(spec, weights, self.device)
        except (ImportError, AttributeError, KeyError, TypeError, OSError) as e:
            raise ValueError(f"Cannot load model {spec!r}: {type(e).__name__}: {e}") from e
        report(0.08, "Build input")
        x, sample_meta = self.build_input(loaded, sample)
        return self._create(loaded, x, sample_meta, modules, max_capture_mb, progress, backend)

    def create_from_objects(
        self,
        model: torch.nn.Module,
        inputs: Any,
        name: str | None = None,
        categories: list[str] | None = None,
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        progress: Progress | None = None,
        tokenizer: Any = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """``tokenizer`` (encode / decode) labels token ids and, for a causal LM, enables the logit lens."""
        model = _prepare_hf(model) if is_hf(model) else model.eval()
        v = vocab_size(model) if tokenizer is not None and categories is None else None
        cats = Vocab(tokenizer, v) if v else categories
        lm = tokenizer is not None and is_lm(model)
        loaded = LoadedModel(model, None, None, categories=cats, tokenizer=tokenizer, lm=lm)
        sample_meta = {"kind": "tensor", "name": name or "in-process input"}
        return self._create(loaded, inputs, sample_meta, modules, max_capture_mb, progress, backend)

    def start_capture(
        self,
        spec: str,
        weights: str | None,
        sample: dict[str, Any],
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        backend: str | None = None,
    ) -> Job:
        """Queue a capture job (its result is the run meta) after the checks that need no model."""
        check_spec(spec)
        uid = Path(sample.get("upload_id") or "").name
        if sample.get("kind") == "upload" and not (uid and (self.uploads / uid).is_file()):
            raise ValueError(f"Upload {sample.get('upload_id')} not found; upload the file again")
        if sample.get("kind") == "text" and not sample.get("text"):
            raise ValueError("Text sample needs text")
        _check_backend(backend)
        run = lambda p: self.create_from_spec(spec, weights, sample, modules, max_capture_mb, p, backend)  # noqa: E731
        return self.jobs.submit("capture", run)

    def _create(
        self,
        loaded: LoadedModel,
        inputs: Any,
        sample_meta: dict[str, Any],
        modules: list[str] | None = None,
        max_capture_mb: float | None = None,
        progress: Progress | None = None,
        backend: str | None = None,
    ) -> dict[str, Any]:
        modules = [m.strip() for m in modules or [] if m.strip()] or None
        if max_capture_mb is not None and not (math.isfinite(max_capture_mb) and max_capture_mb > 0):
            raise ValueError(f"max_capture_mb must be a positive finite number, got {max_capture_mb}")
        budget = int(max_capture_mb * 1024 * 1024) if max_capture_mb else self.max_capture_bytes
        backend = self._pick_backend(loaded.model, backend, progress or _quiet)
        cap = capture if backend == "torchlens" else _module(backend).capture
        sample_meta = {**sample_meta, **_token_meta(loaded, inputs)}
        with self._lock:
            run_id = self.store.new_run_id()
            run_dir = self.store.create(run_id)
            try:
                inputs = _to_device(inputs, self.device)
                ir = cap(loaded.model, inputs, self.store, run_id, budget, modules, progress)
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
                "model": {"spec": loaded.spec, "weights": loaded.weights, "backend": backend,
                          "granularity": "op", **ir["model"], "lm": loaded.lm},  # fmt: skip
                "sample": sample_meta,
                "input_shape": _shape(inputs),
                "output": top,
                "output_error": output_error,
                "preview": preview,
                "categories": loaded.categories if isinstance(loaded.categories, list) else None,
                "reloadable": loaded.spec is not None,
                "status": "ready",
            }
            self.store.write_json(run_id, "ir.json", ir)
            self.store.write_json(run_id, "meta.json", meta)
            self._keep_live(run_id, Live(loaded, inputs))
            return meta

    def _pick_backend(self, model: torch.nn.Module, backend: str | None, report: Progress) -> str:
        """The explicit backend, else NNsight for models above ``nnsight_params`` when it is installed."""
        _check_backend(backend)
        if backend:
            return backend
        n = sum(p.numel() for p in model.parameters())
        if n > self.nnsight_params and _module("nnsight").available():
            report(0.09, f"Large model ({n / 1e9:.1f} B params) · NNsight module capture")
            return "nnsight"
        return "torchlens"

    def _backend(self, run_id: str) -> Any:
        """The capture module that recorded this run (op_gradients and intervene must match its op ids)."""
        return _module(self.store.meta(run_id)["model"].get("backend") or "torchlens")

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
        """The cached result, or None when missing or stale: attribution files before M3 have no merges, and
        before v0.2 an integer-input run got no ``tokens``."""
        try:
            data = self.store.read_json(run_id, name)
        except KeyError:
            return None
        if name.split("/")[-1].split("_")[0] in METHODS:
            old = "tokens" not in data and data.get("input") is None and self._int_input(run_id)
            if "merges" not in data or old:
                return None
        return data

    def _int_input(self, run_id: str) -> bool:
        return any(o["is_input"] and "int" in (o.get("dtype") or "") for o in self.store.ir(run_id)["ops"])

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
                _baseline_id(live.loaded.tokenizer),
                self._backend(run_id).op_gradients,
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
        basis = "|attribution| · 24×24 pooled input map"
        if not maps:
            maps = {m: np.asarray(r["tokens"]["scores"]) for m, r in res.items() if r.get("tokens")}
            basis = "|attribution| · per token"
        block = agreement(maps, METHODS, basis) if len(maps) > 1 else None
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

    # ---- interventions (v0.3) ---------------------------------------------
    def _source(self, run_id: str, src: str, op: str, shape: list[int] | None = None) -> np.ndarray:
        """``src``'s saved tensor of ``op`` after the source-run checks (same model spec and weights)."""
        if not self.store.exists(src):
            raise ValueError(f"Source run {src} not found")
        mine, theirs = (self.store.meta(r)["model"] for r in (run_id, src))
        if src != run_id and (mine.get("spec") is None or _model_key(mine) != _model_key(theirs)):
            raise ValueError(
                f"{src} is a run of another model ({theirs.get('spec')} · {theirs.get('weights')}); "
                "use a run with the same model spec and weights"
            )
        try:
            arr = np.array(self.store.load_tensor(src, op))
        except KeyError:
            raise ValueError(f"{src} has no saved tensor for {op}") from None
        if shape is not None and list(arr.shape) != list(shape):
            raise ValueError(f"{src}'s {op} has shape {list(arr.shape)}, this run's {list(shape)}")
        return arr

    def _peers(self, run_id: str, op: str, shape: list[int]) -> list[str]:
        """Runs of this model (this one included) with a saved ``op`` tensor of ``shape``, sorted."""
        mine = self.store.meta(run_id)["model"]
        out = []
        for meta in self.store.list_runs():
            if meta["id"] != run_id and (
                mine.get("spec") is None or _model_key(meta["model"]) != _model_key(mine)
            ):
                continue
            path = self.store.tensor_path(meta["id"], op)
            if path.exists() and list(np.load(path, mmap_mode="r").shape) == list(shape):
                out.append(meta["id"])
        return sorted(out)

    def resolve(
        self, run_id: str, norm: dict[str, Any]
    ) -> list[tuple[np.ndarray | None, str | None, int | None]]:
        """Per edit (value, resolved source run, runs averaged): the replacement or steer vector read from the
        store, after the source-run checks of spec §1.3 (400 at POST)."""
        shapes = {o["id"]: o["shape"] for o in self.store.ir(run_id)["ops"]}
        out: list[tuple[np.ndarray | None, str | None, int | None]] = []
        for e in norm["edits"]:
            op, kind = e["op"], e["kind"]
            shape = shapes[op]
            if kind == "resample" and e["source_run"] == "random":
                peers = [r for r in self._peers(run_id, op, shape) if r != run_id]
                if not peers:
                    raise ValueError(f"No other run of this model has a saved tensor for {op}")
                src = random.Random(e["seed"]).choice(peers)
                out.append((self._source(run_id, src, op, shape), src, None))
            elif kind == "resample":
                out.append((self._source(run_id, e["source_run"], op, shape), e["source_run"], None))
            elif kind == "mean" and e["over"] == "runs":
                peers = self._peers(run_id, op, shape)
                if len(peers) < 2:
                    raise ValueError(
                        f"Mean over runs needs another saved run of this model with a tensor for {op}"
                    )
                total = np.zeros(shape)
                for r in peers:
                    total += self.store.load_tensor(r, op)
                out.append(((total / len(peers)).astype(np.float32), None, len(peers)))
            elif kind == "steer" and e["direction"]["kind"] == "contrast":
                d, n = e["direction"], shape[CH_AXIS[len(shape)]]
                va, vb = (_pool(self._source(run_id, d[k], op), d["pool"]) for k in ("run_a", "run_b"))
                if not len(va) == len(vb) == n:
                    raise ValueError(f"Contrast runs have {len(va)} and {len(vb)} channels at {op}")
                out.append(((vb - va).astype(np.float32), None, None))
            else:
                out.append((None, None, None))
        return out

    def _labels(self, ir: dict[str, Any], norm: dict[str, Any], resolved: list) -> list[str]:
        return [
            label(e, _path(ir, e["op"]), src, n)
            for e, (_, src, n) in zip(norm["edits"], resolved, strict=True)
        ]

    def intervene(self, run_id: str, norm: dict[str, Any], resolved: list | None = None,
                  progress: Progress | None = None) -> dict[str, Any]:  # fmt: skip
        """Run a normalised intervention set (one forward pass), save the downstream tensors under its hash
        and return the InterventionResult (cached in ``interventions/<hash>/meta.json``)."""
        h = set_hash(norm)
        name = f"interventions/{h}/meta.json"
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        report = progress or _quiet
        resolved = self.resolve(run_id, norm) if resolved is None else resolved
        ir = self.store.ir(run_id)
        ops = {e["op"] for e in norm["edits"]}
        edited = [o["id"] for o in ir["ops"] if o["id"] in ops]
        down = downstream(ir, edited)
        budget = ir["model"].get("budget_bytes") or self.max_capture_bytes
        saved, truncated = keep_ops(ir, down, budget)
        edits = [(e["op"], edit_fn(e, v)) for e, (v, _, _) in zip(norm["edits"], resolved, strict=True)]
        backend = self.store.meta(run_id)["model"].get("backend") or "torchlens"
        with self._lock:
            live = self.live(run_id)
            model = live.loaded.model.eval()
            report(0.05, "Clean pass")
            # grad mode as at capture: under no_grad nn.MultiheadAttention takes its fused path (other op ids)
            logits = to_logits(model(*_args(live.inputs)))[0].detach().float()
            t = int(logits.argmax()) if norm["target"] is None else norm["target"]
            if not 0 <= t < logits.numel():
                raise ValueError(f"Target {t} is outside the output (0–{logits.numel() - 1})")
            report(0.15, "Intervene")
            out = self._backend(run_id).intervene(model, live.inputs, edits, down, saved)
            after = to_logits(out["logits_after"])[0].detach().float()
        kept = [op for op in saved if op in out["after"]]
        nbytes = 0
        for i, op in enumerate(kept, start=1):
            report(0.8 + 0.19 * i / len(kept), f"Save tensors {i}/{len(kept)}")
            self.store.save_tensor(run_id, op, out["after"][op], iv=h)
            nbytes += out["after"][op].nbytes
        cats = live.loaded.categories
        p0, p1 = torch.softmax(logits, 0), torch.softmax(after, 0)
        before_ = {"logit": _fin(logits[t]), "prob": _fin(p0[t])}
        after_ = {"logit": _fin(after[t]), "prob": _fin(p1[t])}
        delta = {k: None if (a := after_[k]) is None or (b := before_[k]) is None else a - b for k in before_}
        labels = self._labels(ir, norm, resolved)
        result = {
            "hash": h,
            "run_id": run_id,
            "name": labels[0] + (f" (+{len(labels) - 1})" if len(labels) > 1 else ""),
            "created": self.store.now(),
            "set": norm,
            "labels": labels,
            "method_label": IV_LABELS.get(backend, IV_LABELS["torchlens"]),
            "target": t,
            "target_label": cats[t] if cats and t < len(cats) else None,
            "before": before_,
            "after": after_,
            "delta_logit": delta["logit"],
            "delta_prob": delta["prob"],
            "top_before": _top_of(logits, cats),
            "top_after": _top_of(after, cats),
            "edited": edited,
            "downstream": down,
            "saved": kept,
            "truncated": truncated,
            "budget_bytes": budget,
            "bytes": nbytes,
            "nodes": out["nodes"],
            "stats": out["stats"],
        }
        self.store.write_json(run_id, name, result)
        return result

    @staticmethod
    def _gen_sites(meta: dict[str, Any], ir: dict[str, Any], norm: dict[str, Any]) -> list[tuple[str, int]]:
        """The (module, call) of every edit, after the generation checks that need no model."""
        if not meta["model"].get("lm"):
            raise ValueError(NOT_LM_GEN)
        if not meta["sample"].get("token_ids"):
            raise ValueError("Generation needs token ids as input")
        sites = []
        for e in norm["edits"]:
            op, kind = e["op"], e["kind"]
            if kind not in GEN_KINDS and not (kind == "mean" and e["over"] == "positions"):
                what = "mean over runs" if kind == "mean" else kind
                raise ValueError(
                    f"Generation supports zero, mean over positions, constant, scale and steer edits "
                    f"({what} values have the prompt's length)"
                )
            site = module_site(ir, op)
            if site is None:
                raise ValueError(
                    f"Generation edits {op} through a forward hook on its module; {op} is not the "
                    "output of a module (pick its module node)"
                )
            sites.append(site)
        return sites

    def generate(self, run_id: str, norm: dict[str, Any], n: int, resolved: list | None = None,
                 progress: Progress | None = None) -> dict[str, Any]:  # fmt: skip
        """Greedy continuations of the run's prompt, clean and with the set's edits as forward hooks
        (GenerationResult, cached in ``interventions/<hash>/generate_<n>.json``)."""
        h = set_hash(norm)
        name = f"interventions/{h}/generate_{n}.json"
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        meta, ir = self.store.meta(run_id), self.store.ir(run_id)
        sites = self._gen_sites(meta, ir, norm)
        if not 1 <= n <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
        resolved = self.resolve(run_id, norm) if resolved is None else resolved
        hooks = [(s, edit_fn(e, v)) for s, e, (v, _, _) in zip(sites, norm["edits"], resolved, strict=True)]
        with self._lock:
            live = self.live(run_id)
            model = live.loaded.model.eval()
            x = _first(live.inputs)
            cfg = getattr(model, "config", None)
            limit = getattr(cfg, "n_positions", 0) or getattr(cfg, "max_position_embeddings", 0)
            if limit and x.shape[-1] + n > limit:
                raise ValueError(
                    f"The prompt ({x.shape[-1]} tokens) plus {n} new tokens exceeds the model's "
                    f"{limit} positions"
                )
            res = generate_tokens(model, x, hooks, n, live.loaded.tokenizer, progress)
        result = {
            "hash": h,
            "n": n,
            "labels": self._labels(ir, norm, resolved),
            "method_label": f"Greedy decoding · {n} new tokens · edits as forward hooks on module outputs · "
            "no KV cache",  # fmt: skip
            "prompt": meta["sample"].get("tokens") or [],
            **res,
        }
        self.store.write_json(run_id, name, result)
        return result

    def interventions(self, run_id: str) -> list[dict[str, Any]]:
        """InterventionSummary rows, newest first (folders without meta.json are generate-only: skipped)."""
        root = self.store.run_dir(run_id) / "interventions"
        rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/meta.json"))] if root.is_dir() else []
        return sorted(
            ({k: r.get(k) for k in SUMMARY} for r in rows), key=lambda r: r["created"] or 0, reverse=True
        )

    def intervention(self, run_id: str, h: str) -> dict[str, Any]:
        try:
            self.store.iv_dir(run_id, h)
            return self.store.read_json(run_id, f"interventions/{h}/meta.json")
        except KeyError:
            raise NotFound(f"Intervention {h} not found in {run_id}") from None

    def rename(self, run_id: str, h: str, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be empty")
        res = self.intervention(run_id, h) | {"name": name[:80]}
        self.store.write_json(run_id, f"interventions/{h}/meta.json", res)
        return res

    def delete_intervention(self, run_id: str, h: str) -> None:
        self.intervention(run_id, h)
        shutil.rmtree(self.store.iv_dir(run_id, h), ignore_errors=True)

    def iv_tensor(self, run_id: str, h: str, op: str) -> np.ndarray:
        """``op``'s tensor under intervention ``h``: intervened when saved, clean when not downstream
        (identical), else NotFound with the reason; a clean miss is the store's plain KeyError."""
        res = self.intervention(run_id, h)
        if op in res["saved"]:
            return self.store.load_tensor(run_id, op, iv=h)
        if op not in res["downstream"]:
            return self.store.load_tensor(run_id, op)
        o = next((o for o in self.store.ir(run_id)["ops"] if o["id"] == op), {})
        if o.get("saved") and "float" in (o.get("dtype") or ""):
            mb = (res.get("budget_bytes") or 0) / 2**20
            why = f"the capture budget ({mb:g} MB) was reached before it; the graph still shows its Δ"
        else:
            why = "the clean run saved no tensor for it"
        raise NotFound(f"No intervened tensor saved for {op}: {why}")

    def iv_ir(self, run_id: str, h: str) -> dict[str, Any]:
        """The IR with the intervened stats of ``h`` (for ``/graph?iv=``)."""
        stats = self.intervention(run_id, h)["stats"]
        ir = self.store.ir(run_id)
        for o in ir["ops"]:
            o["stats"] = stats.get(o["id"], o.get("stats"))
        return ir

    def _iv_job(self, run_id: str, kind: str, req: dict[str, Any]) -> Job:
        raw = req.get("interventions")
        if not raw:
            raise ValueError(f"{kind} jobs need interventions")
        meta, ir = self.store.meta(run_id), self.store.ir(run_id)
        norm = normalise(raw, ir)
        n = int(req.get("max_new_tokens") or 16)
        if kind == "generate":
            if not 1 <= n <= MAX_NEW_TOKENS:
                raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
            self._gen_sites(meta, ir, norm)
        name = f"interventions/{set_hash(norm)}/{f'generate_{n}' if kind == 'generate' else 'meta'}.json"
        if (cached := self._cached(run_id, name)) is not None:
            return self.jobs.finished(kind, cached, run_id)
        resolved = self.resolve(run_id, norm)
        if kind == "generate":
            fn = lambda p: self.generate(run_id, norm, n, resolved, p)  # noqa: E731
        else:
            fn = lambda p: self.intervene(run_id, norm, resolved, p)  # noqa: E731
        return self.jobs.submit(kind, fn, run_id, key=name)

    def lens(self, run_id: str, target: int | None, progress: Progress | None = None) -> dict[str, Any]:
        """Logit lens (``lens.logit_lens``) of an LM run; ``target`` None = the top prediction."""
        name = lens_name(target)
        if (cached := self._cached(run_id, name)) is not None:
            return cached
        with self._lock:
            live = self.live(run_id)
            if not live.loaded.lm:
                raise ValueError(NOT_LM)
            x = _first(live.inputs)
            if not (_token_ids(x) and x.shape[0] == 1):
                raise ValueError("Logit lens needs token ids as input")
            result = logit_lens(live.loaded.model, x, target, live.loaded.tokenizer, progress=progress)
        self.store.write_json(run_id, name, result)
        return result

    def job(self, run_id: str, req: dict[str, Any]) -> Job:
        """Start one job, or answer it at once from the cache. ValueError for a bad request (HTTP 400)."""
        meta = self.store.meta(run_id)
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
        elif kind in ("intervene", "generate"):
            return self._iv_job(run_id, kind, req)
        elif kind == "lens":
            if not meta["model"].get("lm"):
                raise ValueError(NOT_LM)
            name = lens_name(target)
            fn = lambda p: self.lens(run_id, target, p)  # noqa: E731
        else:
            raise ValueError(f"Unknown job kind {kind!r}")
        if (cached := self._cached(run_id, name)) is not None:
            return self.jobs.finished(kind, cached, run_id)
        return self.jobs.submit(kind, fn, run_id, key=name)


UNIT_WHY = "unit objectives hook module outputs"
NOT_LM = "Logit lens needs a causal language model with a tokenizer; this run's model is not one."
NOT_LM_GEN = "Generation needs a causal language model with a tokenizer; this run's model is not one."
IV_LABELS = {"torchlens": "Intervention · TorchLens capture-time hooks (every op)",
             "nnsight": "Intervention · NNsight (module outputs)"}  # fmt: skip
SUMMARY = (
    "hash",
    "name",
    "created",
    "labels",
    "target",
    "target_label",
    "delta_logit",
    "delta_prob",
    "truncated",
)


class NotFound(KeyError):
    """A missing intervention or intervened tensor; ``args[0]`` is the 404 text."""


BACKENDS = ("torchlens", "nnsight")


def _module(backend: str) -> Any:
    """The capture module of a backend, imported lazily (only capture.py / capture_nnsight.py trace)."""
    return importlib.import_module(".capture_nnsight" if backend == "nnsight" else ".capture", __package__)


def _check_backend(backend: str | None) -> None:
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, or null for auto")
    if backend == "nnsight" and not (mod := _module("nnsight")).available():
        raise ValueError(mod.MISSING)


def lens_name(target: int | None) -> str:
    return f"lens/{'top' if target is None else int(target)}.json"


def _baseline_id(tok: Any) -> int:
    """IG baseline token for token inputs: the pad token, else end-of-sequence, else 0."""
    for attr in ("pad_token_id", "eos_token_id"):
        if isinstance(i := getattr(tok, attr, None), int):
            return i
    return 0


def _token_ids(x: Any) -> bool:
    """A (B, T) integer tensor."""
    return torch.is_tensor(x) and x.dim() == 2 and x.dtype in (torch.int64, torch.int32)


def _token_meta(loaded: LoadedModel, inputs: Any) -> dict[str, Any]:
    """``token_ids`` and ``tokens`` (one decode per id) for a (1, T) id input of a model with a tokenizer."""
    x = _first(inputs)
    if loaded.tokenizer is None or not _token_ids(x) or x.shape[0] != 1:
        return {}
    ids = [int(i) for i in x[0].tolist()]
    return {"token_ids": ids, "tokens": [loaded.tokenizer.decode([i]) for i in ids]}


def _text_input(loaded: LoadedModel, text: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if loaded.tokenizer is None:
        raise ValueError("This model has no tokenizer; upload a .npy of token ids or a tensor instead.")
    ids = list(loaded.tokenizer.encode(text))
    if not ids:
        raise ValueError("The text has no tokens")
    cfg = getattr(loaded.model, "config", None)
    limit = min(512, getattr(cfg, "n_positions", 0) or getattr(cfg, "max_position_embeddings", 0) or 512)
    if len(ids) > limit:
        raise ValueError(f"The text is {len(ids)} tokens; this model takes at most {limit}")
    return torch.tensor([ids], dtype=torch.int64), {"kind": "text", "name": text[:60], "text": text}


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
) -> str:
    """The run-folder file that caches one job's result; also the job's dedup key."""
    k, s = okey(target, objective), safe_name(op or "")
    if kind == "attribution":
        return f"attribution/{method}_{k}_{steps}.json"
    if kind == "layer":
        return f"attribution/{method}_{k}_{s}_{steps}.json"
    if kind == "neuron":
        return f"attribution/neuron_{k}_{s}_c{channel}_{steps}.json"
    return f"attribution/agreement_{k}_{steps}{'_' + s if op else ''}.json"


def _first(inputs: Any) -> Any:
    return inputs[0] if isinstance(inputs, (tuple, list)) else inputs


def _args(inputs: Any) -> tuple:
    return tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)


def _model_key(model: dict[str, Any]) -> tuple:
    return model.get("spec"), model.get("weights")


def _path(ir: dict[str, Any], op: str) -> str:
    """An edit's label path: its module (``@call`` after the first call), else the op id."""
    site = module_site(ir, op)
    return op if site is None else site[0] + (f"@{site[1]}" if site[1] > 1 else "")


def _pool(a: np.ndarray, pool: str) -> np.ndarray:
    """A saved activation pooled to its channel vector (mean over the other axes; "last" = the last token)."""
    ax = CH_AXIS.get(a.ndim)
    if ax is None:
        return np.zeros(0)
    if a.ndim == 3 and pool == "last":
        a = a[:, -1:, :]
    return a.mean(axis=tuple(i for i in range(a.ndim) if i != ax), dtype=np.float64)


def _float_input(live: Live) -> Live:
    """``live`` when its input is a float tensor: layer methods interpolate from a zero baseline."""
    x = _first(live.inputs)
    if not (torch.is_tensor(x) and x.is_floating_point()):
        raise ValueError(
            "Layer methods interpolate the input from a zero baseline and need a float input; this run's "
            f"input is {getattr(x, 'dtype', type(x).__name__)}; token inputs get Layer IG on the embeddings "
            "in the saliency panel instead."
        )
    return live


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


def _fin(x: Any) -> float | None:
    """A float, or None when non-finite (an edit can overflow the logits)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _top_of(logits: torch.Tensor, cats: Sequence[str] | None, k: int = 5) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=0)
    vals, idx = probs.topk(min(k, probs.numel()))
    return [
        {
            "index": int(i),
            "prob": _fin(v),
            "logit": _fin(logits[i]),
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
