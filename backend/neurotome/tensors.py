"""Tensor views: turn one saved activation into small, browser-sized summaries.

The browser never receives a whole tensor. Each view picks a rendering from the
shape (after dropping the batch axis):

* ``image``    (C, H, W)  channel grid, one channel as a dot matrix, row profile,
                          channel vector at a pixel, PCA of spatial positions
* ``sequence`` (T, D)     token x feature heatmap, per-token norm, PCA of tokens
* ``vector``   (D,)       1D signal and histogram
* ``attention`` (H, T, T) rows are distributions over keys (softmax): per-head maps and pattern summary

Higher-rank tensors are folded into ``image`` (leading axes merged into channels).
Image maps are scaled per channel or on one global range (``norm``), sequentially to
[0, 1] or diverging to [-1, 1] around 0 (``cmap``). The sequence heatmap is max-abs scaled, or divided by
max(|lo|, |hi|) when a global ``lo``/``hi`` is given (compare mode's shared scale).

Also here: ``input_preview`` (the run's input as a picture) and ``tensor_diff``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def safe_range(lo: float, hi: float) -> tuple[float, float]:
    """A histogram range numpy accepts even for (near-)constant tensors."""
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-6 * max(1.0, abs(lo), abs(hi)):
        pad = 0.5 * max(1.0, abs(lo)) * 1e-3 + 1e-6
        return lo - pad, hi + pad
    return lo, hi


def _pool1d(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] <= n:
        return np.asarray(x, dtype=np.float32)
    return np.stack([c.mean(axis=0) for c in np.array_split(np.asarray(x, dtype=np.float32), n, axis=0)])


def pool2d(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """Block-average a 2D array down to at most (h, w)."""
    x = _pool1d(np.asarray(x, dtype=np.float32), h)
    return _pool1d(x.T, w).T


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def _finite_range(x: np.ndarray) -> list[float]:
    f = np.asarray(x)[np.isfinite(x)]
    return [float(f.min()), float(f.max())] if f.size else [0.0, 0.0]


def _scale(g: np.ndarray, lo: float, hi: float, cmap: str) -> np.ndarray:
    """seq: [lo, hi] -> [0, 1]; div: divide by max(|lo|, |hi|) into [-1, 1], so 0 stays 0."""
    with np.errstate(invalid="ignore"):
        if cmap == "div":
            m = max(abs(lo), abs(hi))
            out = np.clip(g / m, -1, 1) if m > 0 else np.zeros_like(g)
        else:
            out = np.clip((g - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(g)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0 if cmap == "div" else 0.0)


def _signed(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x / m if m > 0 else x


def _r(x: np.ndarray, nd: int = 3) -> list:
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def _hist(x: np.ndarray, bins: int = 24) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float32).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"counts": [0] * bins, "range": [0, 1]}
    lo, hi = float(x.min()), float(x.max())
    counts, _ = np.histogram(x.astype(np.float64), bins=bins, range=safe_range(lo, hi))
    return {"counts": counts.astype(int).tolist(), "range": [lo, hi]}


def _pca2(x: np.ndarray, max_points: int = 400) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """PCA of rows of x (N, D) -> (idx, coords (k, 2), explained ratios)."""
    n = x.shape[0]
    idx = np.linspace(0, n - 1, num=min(n, max_points)).astype(int)
    sub = np.nan_to_num(np.asarray(x[idx], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sub = sub - sub.mean(axis=0, keepdims=True)
    if sub.shape[1] < 2 or not np.any(sub):
        return idx, np.zeros((len(idx), 2)), [0.0, 0.0]
    _, s, vt = np.linalg.svd(sub, full_matrices=False)
    coords = sub @ vt[:2].T
    var = s**2
    ratios = (var[:2] / var.sum()).tolist() if var.sum() > 0 else [0.0, 0.0]
    return idx, coords, ratios


def _drop_batch(a: np.ndarray) -> np.ndarray:
    """Remove the batch axis. Usually it is axis 0; some ops run sequence-first, e.g.
    (tokens, 1, dim) inside attention, so a singleton axis elsewhere is dropped instead."""
    if a.ndim < 2:
        return a
    if a.shape[0] == 1:
        return a[0]
    if a.ndim >= 3 and 1 in a.shape[1:]:
        return np.squeeze(a, axis=1 + list(a.shape[1:]).index(1))
    return a[0]  # a real batch: show the first sample


def tensor_view(
    arr: np.ndarray,
    channel: int | None = None,
    pixel: tuple[int, int] | None = None,
    max_channels: int = 32,
    map_size: int = 24,
    dot_size: int = 16,
    norm: str = "channel",
    cmap: str = "seq",
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    shape = list(a.shape)
    a = _drop_batch(a)
    if a.ndim > 3:
        a = a.reshape(-1, *a.shape[-2:])
    if _is_attention(a):
        return _attention_view(a, shape, channel)
    given = None  # the caller's [lo, hi] for norm=global
    if norm == "global" and lo is not None and hi is not None and bool(np.isfinite([lo, hi]).all()):
        given = [float(lo), float(hi)]
    if a.ndim == 3 and a.shape[1] > 1 and a.shape[2] > 1:
        scale = None
        if norm == "global":
            scale = given or _finite_range(a)
        return _image_view(a, shape, channel, pixel, max_channels, map_size, dot_size, scale, cmap) | {
            "norm": norm,
            "cmap": cmap,
            "scale": scale,
        }
    if a.ndim == 3:
        a = a.reshape(a.shape[0], -1)
    if a.ndim == 2:
        return _sequence_view(a, shape, given)
    return _vector_view(a.reshape(-1), shape)


def _image_view(a, shape, channel, pixel, max_channels, map_size, dot_size, scale, cmap) -> dict[str, Any]:
    C, H, W = a.shape
    means = a.reshape(C, -1).mean(axis=1)
    order = np.argsort(-means)[:max_channels]
    sel = int(channel) if channel is not None and 0 <= int(channel) < C else int(order[0])

    def scaled(g: np.ndarray) -> tuple[np.ndarray, list[float]]:
        r = scale or _finite_range(g)
        return _scale(g, r[0], r[1], cmap), r

    maps = []
    for c in order:
        m, r = scaled(pool2d(a[c], map_size, map_size))
        maps.append({"channel": int(c), "mean": float(means[c]), "map": _r(m, 2), "range": r})
    ch = a[sel]
    peak = np.unravel_index(int(np.argmax(ch)), ch.shape)
    py, px = peak if pixel is None else (min(max(pixel[0], 0), H - 1), min(max(pixel[1], 0), W - 1))
    dots, dots_range = scaled(pool2d(ch, dot_size, dot_size))
    profile = _pool1d(ch[py], 128)
    vec = a[:, py, px]
    idx, coords, ratios = _pca2(a.reshape(C, -1).T)
    color = _norm01(ch.reshape(-1)[idx])
    return {
        "kind": "image",
        "shape": shape,
        "channels": C,
        "height": H,
        "width": W,
        "channel": sel,
        "channel_rank": int(np.where(np.argsort(-means) == sel)[0][0]) + 1,
        "maps": maps,
        "dots": _r(dots, 3),
        "dots_range": dots_range,
        "stats": {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "min": float(ch.min()),
            "max": float(ch.max()),
        },
        "peak": [int(peak[0]), int(peak[1])],
        "profile": {"row": int(py), "values": _r(profile)},
        "hist": _hist(ch),
        "vector": {"pixel": [int(py), int(px)], "values": _r(_pool1d(vec, 512))},
        "pca": {"points": _r(coords), "color": _r(color, 2), "explained": _r(np.array(ratios), 3)},
    }


def _is_attention(a: np.ndarray) -> bool:
    """(H, T, T) with every query row a probability distribution over keys (softmax output; bf16 rows sum to
    1 +- ~3e-3)."""
    if a.ndim != 3 or a.shape[1] != a.shape[2] or a.shape[1] <= 1 or not np.isfinite(a).all():
        return False
    return bool(a.min() >= -1e-6 and a.max() <= 1 + 1e-6 and np.abs(a.sum(-1) - 1).max() <= 1e-2)


def _unit_map(g: np.ndarray, nd: int) -> dict[str, Any]:
    m = float(g.max())
    return {"map": _r(g / m if m > 0 else np.zeros_like(g), nd), "max": m}


def _head_summary(h: int, p: np.ndarray) -> dict[str, Any]:
    T = p.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)
    q = np.arange(T)
    s = {
        "head": h,
        "entropy": float(ent.mean()),
        "prev": float(p[q[1:], q[1:] - 1].mean()),
        "first": float(p[:, 0].mean()),
        "self": float(p[q, q].mean()),
    }
    names = (("previous", "prev"), ("first", "first"), ("self", "self"))
    return s | {"pattern": next((n for n, k in names if s[k] > 0.5), None)}


def _attention_view(a: np.ndarray, shape: list[int], channel: int | None) -> dict[str, Any]:
    H, T, _ = a.shape
    head = min(max(int(channel or 0), 0), H - 1)
    return {
        "kind": "attention",
        "shape": shape,
        "heads": H,
        "tokens": T,
        "head": head,
        "maps": [{"head": h} | _unit_map(pool2d(a[h], 64, 64), 2) for h in range(min(H, 64))],
        "full": _unit_map(pool2d(a[head], 128, 128), 3),
        "summary": [_head_summary(h, a[h].astype(np.float64)) for h in range(H)],
        "stats": {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        },
        "hist": _hist(a[head]),
    }


def _sequence_view(a, shape, scale=None) -> dict[str, Any]:
    T, D = a.shape
    pooled = pool2d(a, 64, 128)
    heat = _signed(pooled) if scale is None else _scale(pooled, scale[0], scale[1], "div")
    norms = np.linalg.norm(a, axis=1)
    idx, coords, ratios = _pca2(a)
    return {
        "kind": "sequence",
        "shape": shape,
        "tokens": T,
        "features": D,
        "heatmap": _r(heat, 3),
        "scale": scale,
        "token_norms": _r(norms),
        "hist": _hist(a),
        "stats": {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        },
        "pca": {
            "points": _r(coords),
            "color": _r(_norm01(norms[idx]), 2),
            "explained": _r(np.array(ratios), 3),
        },
    }


def _vector_view(a, shape) -> dict[str, Any]:
    top = np.argsort(-np.abs(a))[:10]
    return {
        "kind": "vector",
        "shape": shape,
        "length": int(a.shape[0]),
        "values": _r(_pool1d(a, 512)),
        "top": [{"index": int(i), "value": float(a[i])} for i in top],
        "hist": _hist(a),
        "stats": {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        },
    }


def input_preview(
    x: np.ndarray, norm: tuple[Sequence[float], Sequence[float]] | None = None
) -> np.ndarray | None:
    """Sample 0 of an image batch (N, 1|3, H, W) as uint8 (H, W, 3) or (H, W); None for anything else.

    With ``norm = (mean, std)`` the input is de-normalised (x*std + mean, clipped to [0, 1]); without it
    each channel is min-max stretched.
    """
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1] not in (1, 3) or min(x.shape[2:]) < 8:
        return None
    img = np.nan_to_num(x[0].astype(np.float64))
    C = img.shape[0]
    if norm is not None and len(norm[0]) == C and len(norm[1]) == C:
        mean, std = (np.asarray(v, dtype=np.float64).reshape(C, 1, 1) for v in norm)
        img = np.clip(img * std + mean, 0, 1)
    else:
        lo, hi = img.min(axis=(1, 2), keepdims=True), img.max(axis=(1, 2), keepdims=True)
        img = np.where(hi > lo, (img - lo) / np.where(hi > lo, hi - lo, 1), 0)
    out = np.round(img * 255).astype(np.uint8)
    return out[0] if C == 1 else out.transpose(1, 2, 0)


def _num(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def tensor_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Statistics of b - a over positions where both are finite; ``diff`` is None when the shapes differ."""
    out = {"shape_a": list(a.shape), "shape_b": list(b.shape), "same_shape": a.shape == b.shape, "diff": None}
    if a.shape != b.shape:
        return out
    # ponytail: whole-array float64, chunk it if a run's tensors outgrow RAM
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = y - x
    n = d.size > 0
    na, nb, nd = (float(np.linalg.norm(v)) for v in (x, y, d))
    out["diff"] = {
        "stats": {
            k: _num(f(d)) if n else None
            for k, f in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max))
        },
        "hist": _hist(d),
        "max_abs": _num(np.abs(d).max()) if n else None,
        "mean_abs": _num(np.abs(d).mean()) if n else None,
        "cosine": _num(np.dot(x, y) / (na * nb)) if n and na > 0 and nb > 0 else None,
        "rel_l2": _num(nd / na) if n and na > 0 else None,
    }
    return out
