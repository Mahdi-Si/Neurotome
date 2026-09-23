"""NumPy-style slices of a saved tensor, parsed with regexes only (never eval).

Grammar: optional outer ``[ ]``, comma-separated items, each ``...``, an index (``-1``) or
``start:stop:step`` with any part optional. Missing trailing axes mean ``:``.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

SLICE_CAP = 4096  # values in one JSON table response
NPY_CAP = 1 << 24  # values in one .npy download (64 MB float32)

_INDEX = re.compile(r"-?\d+")
_SLICE = re.compile(r"(-?\d+)?:(-?\d+)?(?::(-?\d+)?)?")


class SliceError(ValueError):
    """Bad syntax or an index out of range (HTTP 400)."""


class SliceTooLarge(ValueError):
    """More values than the .npy cap (HTTP 413)."""


def _items(text: str) -> list[str]:
    if len(text) > 200:
        raise SliceError("Slice expression is too long (max 200 characters).")
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
    return [i.strip() for i in t.split(",")] if t else []


def parse_slice(text: str, shape: Sequence[int]) -> tuple:
    """A full-length key of ints and slices for ``arr[key]``."""
    parsed: list[Any] = []
    for n, item in enumerate(_items(text), 1):
        if item == "...":
            if Ellipsis in parsed:
                raise SliceError('Only one "..." is allowed.')
            parsed.append(Ellipsis)
        elif _INDEX.fullmatch(item):
            parsed.append(int(item))
        elif m := _SLICE.fullmatch(item):
            parsed.append(slice(*(int(g) if g else None for g in m.groups())))
        else:
            raise SliceError(f'Cannot read "{item}" (item {n}). Use integers, start:stop:step, ":" or "...".')
    count = len(parsed) - (Ellipsis in parsed)
    fill = len(shape) - count
    if fill < 0:
        dims = "×".join(map(str, shape))
        raise SliceError(f"Too many indices: {count} for a {len(shape)}-D tensor ({dims}).")
    at = parsed.index(Ellipsis) if Ellipsis in parsed else len(parsed)
    key = parsed[:at] + [slice(None)] * fill + parsed[at + 1 :]
    for axis, (k, dim) in enumerate(zip(key, shape, strict=True)):
        if isinstance(k, slice):
            if k.step == 0:
                raise SliceError(f"Slice step cannot be zero (axis {axis}).")
        elif not -dim <= k < dim:
            raise SliceError(f"Index {k} is out of range for axis {axis} (size {dim}).")
        else:
            key[axis] = k % dim
    return tuple(key)


def _fmt(k: int | slice, dim: int) -> str:
    if isinstance(k, int):
        return str(k)
    start, stop, step = k.indices(dim)
    return f"{start}:{'' if stop < 0 else stop}:{step}"


def _value(v: float) -> float | str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(f"{v:.6g}")


def slice_table(arr: np.ndarray, text: str) -> dict[str, Any]:
    """The slice as a flat C-order table, truncated from the last axis backwards to <= SLICE_CAP values."""
    key = parse_slice(text, arr.shape)
    view = np.asarray(arr[key])
    axes = [i for i, k in enumerate(key) if isinstance(k, slice)]
    kept: list[int] = []
    for dim in reversed(view.shape):
        kept.insert(0, min(dim, max(1, SLICE_CAP // max(1, math.prod(kept)))))
    sub = view[tuple(slice(0, k) for k in kept)]
    return {
        "op": None,
        "expr": text,
        "key": "[" + ", ".join(_fmt(k, d) for k, d in zip(key, arr.shape, strict=True)) + "]",
        "shape": kept,
        "full_shape": list(view.shape),
        "truncated": kept != list(view.shape),
        "axes": axes,
        "indices": [list(range(*key[a].indices(arr.shape[a])))[:k] for a, k in zip(axes, kept, strict=True)],
        "values": [_value(v) for v in np.asarray(sub, dtype=np.float64).ravel().tolist()],
    }


def slice_npy(arr: np.ndarray, text: str, cap: int | None = None) -> bytes:
    """``np.save`` bytes of the whole slice (float32); SliceTooLarge above ``cap`` values (NPY_CAP)."""
    cap = NPY_CAP if cap is None else cap
    key = parse_slice(text, arr.shape)
    n = math.prod(len(range(*k.indices(d))) for k, d in zip(key, arr.shape, strict=True) if type(k) is slice)
    if n > cap:
        raise SliceTooLarge(
            f"Slice has {n:,} values; .npy export is capped at {cap:,} ({cap * 4 / 2**20:.3g} MB float32). "
            "Narrow the slice."
        )
    buf = io.BytesIO()
    np.save(buf, np.array(arr[key], dtype=np.float32, order="C"))
    return buf.getvalue()
