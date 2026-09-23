"""On-disk run store.

Layout of one run::

    runs/<run_id>/
      meta.json          run summary (model spec, sample, output top-k, sizes)
      ir.json            the captured graph (see capture.py for the schema)
      inputs.pt          the exact input tensors, so attribution can re-run later
      input_preview.png  the input as a picture (<= 224 px), only for image-like inputs (absent before M2)
      tensors/<op>.npy   one activation per op, float32, memory-mapped on read
      attribution/*.json cached attribution results
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")
PREVIEW = "input_preview.png"


def safe_name(op_id: str) -> str:
    return _SAFE.sub("_", op_id)


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- runs -------------------------------------------------------------
    def new_run_id(self) -> str:
        existing = [p.name for p in self.root.glob("run_*") if p.is_dir()]
        nums = [int(n[4:]) for n in existing if n[4:].isdigit()]
        return f"run_{(max(nums) + 1) if nums else 1:04d}"

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"run_\d{4,}", run_id):
            raise KeyError(run_id)
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        d = self.run_dir(run_id)
        (d / "tensors").mkdir(parents=True, exist_ok=False)
        (d / "attribution").mkdir()
        return d

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)

    def exists(self, run_id: str) -> bool:
        try:
            return (self.run_dir(run_id) / "meta.json").exists()
        except KeyError:
            return False

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted(self.root.glob("run_*"), reverse=True):
            meta = d / "meta.json"
            if meta.exists():
                out.append(json.loads(meta.read_text()))
        return out

    # ---- json blobs -------------------------------------------------------
    def write_json(self, run_id: str, name: str, data: Any) -> None:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)

    def read_json(self, run_id: str, name: str) -> Any:
        path = self.run_dir(run_id) / name
        if not path.exists():
            raise KeyError(f"{run_id}/{name}")
        return json.loads(path.read_text())

    def meta(self, run_id: str) -> dict[str, Any]:
        return self.read_json(run_id, "meta.json")

    def ir(self, run_id: str) -> dict[str, Any]:
        return self.read_json(run_id, "ir.json")

    # ---- tensors ----------------------------------------------------------
    def tensor_path(self, run_id: str, op_id: str) -> Path:
        return self.run_dir(run_id) / "tensors" / f"{safe_name(op_id)}.npy"

    def save_tensor(self, run_id: str, op_id: str, array: np.ndarray) -> int:
        path = self.tensor_path(run_id, op_id)
        np.save(path, np.ascontiguousarray(array, dtype=np.float32))
        return path.stat().st_size

    def load_tensor(self, run_id: str, op_id: str) -> np.ndarray:
        path = self.tensor_path(run_id, op_id)
        if not path.exists():
            raise KeyError(f"No saved tensor for {op_id}")
        return np.load(path, mmap_mode="r")

    @staticmethod
    def now() -> float:
        return time.time()
