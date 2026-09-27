"""On-disk run store.

Layout of one run::

    runs/<run_id>/
      meta.json          run summary (model spec, sample, output top-k, sizes)
      ir.json            the captured graph (see capture.py for the schema)
      inputs.pt          the exact input tensors, so attribution can re-run later
      input_preview.png  the input as a picture (<= 224 px), only for image-like inputs (absent before M2)
      tensors/<op>.npy   one activation per op, float32, memory-mapped on read
      attribution/*.json cached attribution results
      interventions/<hash>/{meta.json, tensors/<op>.npy, generate_<n>.json}   (v0.3)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")
PREVIEW = "input_preview.png"


def safe_name(op_id: str) -> str:
    return _SAFE.sub("_", op_id)


def _retry(fn: Any, tries: int = 20) -> Any:
    """``fn()``, retried while Windows refuses a file that another thread is reading or replacing."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.005 * (i + 1))


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
        # numeric order: run_10000 is newer than run_9999
        for d in sorted(self.root.glob("run_*"), key=lambda d: (len(d.name), d.name), reverse=True):
            meta = d / "meta.json"
            if meta.exists():
                out.append(json.loads(_retry(meta.read_text)))
        return out

    # ---- json blobs -------------------------------------------------------
    def _mkdirs(self, run_id: str, path: Path) -> None:
        """Create ``path``'s folders inside the run folder, never the run folder itself: a job that ends after
        its run was deleted fails (FileNotFoundError) instead of leaving a folder without meta.json."""
        d = self.run_dir(run_id)
        for part in path.parent.relative_to(d).parts:
            d = d / part
            d.mkdir(exist_ok=True)
        if not d.is_dir():
            raise FileNotFoundError(f"Run {run_id} was deleted")

    def write_json(self, run_id: str, name: str, data: Any) -> None:
        path = self.run_dir(run_id) / name
        self._mkdirs(run_id, path)
        # one temporary file per writer: two threads writing one cache file never share it
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        _retry(lambda: tmp.replace(path))

    def read_json(self, run_id: str, name: str) -> Any:
        path = self.run_dir(run_id) / name
        if not path.exists():
            raise KeyError(f"{run_id}/{name}")
        return json.loads(_retry(path.read_text))

    def meta(self, run_id: str) -> dict[str, Any]:
        return self.read_json(run_id, "meta.json")

    def ir(self, run_id: str) -> dict[str, Any]:
        return self.read_json(run_id, "ir.json")

    # ---- tensors ----------------------------------------------------------
    def iv_dir(self, run_id: str, iv: str) -> Path:
        """An intervention's folder; KeyError unless ``iv`` is a 12-hex-digit set hash."""
        if not re.fullmatch(r"[0-9a-f]{12}", iv):
            raise KeyError(iv)
        return self.run_dir(run_id) / "interventions" / iv

    def tensor_path(self, run_id: str, op_id: str, iv: str | None = None) -> Path:
        base = self.iv_dir(run_id, iv) if iv else self.run_dir(run_id)
        return base / "tensors" / f"{safe_name(op_id)}.npy"

    def save_tensor(self, run_id: str, op_id: str, array: np.ndarray, iv: str | None = None) -> int:
        path = self.tensor_path(run_id, op_id, iv)
        self._mkdirs(run_id, path)
        np.save(path, np.ascontiguousarray(array, dtype=np.float32))
        return path.stat().st_size

    def load_tensor(self, run_id: str, op_id: str, iv: str | None = None) -> np.ndarray:
        path = self.tensor_path(run_id, op_id, iv)
        if not path.exists():
            raise KeyError(f"No saved tensor for {op_id}")
        return np.load(path, mmap_mode="r")

    @staticmethod
    def now() -> float:
        return time.time()
