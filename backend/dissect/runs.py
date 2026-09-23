"""Run manager: creates runs (load model -> build input -> capture) and keeps live models for attribution."""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .attribution import attribute
from .capture import capture
from .models import LoadedModel, load_model, to_logits
from .store import RunStore


@dataclass
class Live:
    loaded: LoadedModel
    inputs: Any


class RunManager:
    def __init__(self, store: RunStore, device: str = "cpu", max_capture_bytes: int | None = None):
        self.store = store
        self.device = device
        self.max_capture_bytes = max_capture_bytes
        self._live: dict[str, Live] = {}
        self._lock = threading.Lock()  # one capture / attribution at a time (GPU memory)
        self.uploads = store.root / "_uploads"
        self.uploads.mkdir(exist_ok=True)

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
                import numpy as np

                x = torch.from_numpy(np.load(path)).float()
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
    def create_from_spec(self, spec: str, weights: str | None, sample: dict[str, Any]) -> dict[str, Any]:
        loaded = load_model(spec, weights, self.device)
        x, sample_meta = self.build_input(loaded, sample)
        return self._create(loaded, x, sample_meta)

    def create_from_objects(
        self,
        model: torch.nn.Module,
        inputs: Any,
        name: str | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        loaded = LoadedModel(model=model.eval(), spec=None, weights=None, categories=categories)
        sample_meta = {"kind": "tensor", "name": name or "in-process input"}
        return self._create(loaded, inputs, sample_meta)

    def _create(self, loaded: LoadedModel, inputs: Any, sample_meta: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            run_id = self.store.new_run_id()
            run_dir = self.store.create(run_id)
            try:
                inputs = _to_device(inputs, self.device)
                ir = capture(loaded.model, inputs, self.store, run_id, self.max_capture_bytes)
                torch.save(_to_device(inputs, "cpu"), run_dir / "inputs.pt")
                top = _top_k(loaded, inputs)
            except Exception:
                self.store.delete(run_id)
                raise
            meta = {
                "id": run_id,
                "created": self.store.now(),
                "model": {"spec": loaded.spec, "weights": loaded.weights, **ir["model"]},
                "sample": sample_meta,
                "input_shape": _shape(inputs),
                "output": top,
                "categories": loaded.categories,
                "reloadable": loaded.spec is not None,
                "status": "ready",
            }
            self.store.write_json(run_id, "ir.json", ir)
            self.store.write_json(run_id, "meta.json", meta)
            self._live[run_id] = Live(loaded, inputs)
            return meta

    def live(self, run_id: str) -> Live:
        if run_id in self._live:
            return self._live[run_id]
        meta = self.store.meta(run_id)
        if not meta.get("reloadable"):
            raise RuntimeError(
                "This run was created in-process and its model is no longer loaded. "
                "Re-run dissect.open(model, x) to attribute it."
            )
        loaded = load_model(meta["model"]["spec"], meta["model"]["weights"], self.device)
        inputs = torch.load(self.store.run_dir(run_id) / "inputs.pt", map_location=self.device)
        self._live[run_id] = Live(loaded, inputs)
        return self._live[run_id]

    def attribution(self, run_id: str, method: str, target: int | None, steps: int) -> dict[str, Any]:
        name = f"attribution/{method}_{'top' if target is None else target}_{steps}.json"
        try:
            return self.store.read_json(run_id, name)
        except KeyError:
            pass
        with self._lock:
            live = self.live(run_id)
            ir = self.store.ir(run_id)
            unit_ops = {o["id"] for o in ir["ops"] if o["num_params"] > 0 and o["type"] not in ("batchnorm",)}
            result = attribute(
                live.loaded.model,
                live.inputs,
                method,
                target,
                steps,
                live.loaded.categories,
                unit_ops or None,
            )
            where = {o["id"]: o.get("module") or o["id"] for o in ir["ops"]}
            for u in result["top_units"]:
                u["path"] = where.get(u["op"], u["op"])
        self.store.write_json(run_id, name, result)
        return result


def _to_device(inputs: Any, device: str) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, (tuple, list)):
        return type(inputs)(_to_device(i, device) for i in inputs)
    return inputs


def _shape(inputs: Any) -> list[int] | None:
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    return list(x.shape) if torch.is_tensor(x) else None


def _top_k(loaded: LoadedModel, inputs: Any, k: int = 5) -> list[dict[str, Any]]:
    try:
        with torch.no_grad():
            args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
            logits = to_logits(loaded.model(*args))[0].float()
    except Exception:  # noqa: BLE001 - exotic outputs: skip the summary, keep the run
        return []
    probs = torch.softmax(logits, dim=0)
    vals, idx = probs.topk(min(k, probs.numel()))
    cats = loaded.categories
    return [
        {
            "index": int(i),
            "prob": float(v),
            "logit": float(logits[i]),
            "label": cats[int(i)] if cats and int(i) < len(cats) else None,
        }
        for v, i in zip(vals, idx, strict=True)
    ]
