"""Tensor views: turn one saved activation into small, browser-sized summaries.

The browser never receives a whole tensor. Each view picks a rendering from the
shape (after dropping the batch axis):

* ``image``    (C, H, W)  channel grid, one channel as a dot matrix, row profile,
                          channel vector at a pixel, PCA of spatial positions
* ``sequence`` (T, D)     token x feature heatmap, per-token norm, PCA of tokens
* ``vector``   (D,)       1D signal and histogram

Higher-rank tensors are folded into ``image`` (leading axes merged into channels).
"""

from __future__ import annotations

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
    sub = np.asarray(x[idx], dtype=np.float64)
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
) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    shape = list(a.shape)
    a = _drop_batch(a)
    if a.ndim > 3:
        a = a.reshape(-1, *a.shape[-2:])
    if a.ndim == 3 and a.shape[1] > 1 and a.shape[2] > 1:
        return _image_view(a, shape, channel, pixel, max_channels, map_size, dot_size)
    if a.ndim == 3:
        a = a.reshape(a.shape[0], -1)
    if a.ndim == 2:
        return _sequence_view(a, shape)
    return _vector_view(a.reshape(-1), shape)


def _image_view(a, shape, channel, pixel, max_channels, map_size, dot_size) -> dict[str, Any]:
    C, H, W = a.shape
    means = a.reshape(C, -1).mean(axis=1)
    order = np.argsort(-means)[:max_channels]
    sel = int(channel) if channel is not None and 0 <= int(channel) < C else int(order[0])

    maps = [
        {"channel": int(c), "mean": float(means[c]), "map": _r(_norm01(pool2d(a[c], map_size, map_size)), 2)}
        for c in order
    ]
    ch = a[sel]
    py, px = (
        np.unravel_index(int(np.argmax(ch)), ch.shape)
        if pixel is None
        else (min(max(pixel[0], 0), H - 1), min(max(pixel[1], 0), W - 1))
    )
    dots = _norm01(pool2d(ch, dot_size, dot_size))
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
        "stats": {
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "min": float(ch.min()),
            "max": float(ch.max()),
        },
        "peak": [int(py), int(px)],
        "profile": {"row": int(py), "values": _r(profile)},
        "hist": _hist(ch),
        "vector": {"pixel": [int(py), int(px)], "values": _r(_pool1d(vec, 512))},
        "pca": {"points": _r(coords), "color": _r(color, 2), "explained": _r(np.array(ratios), 3)},
    }


def _sequence_view(a, shape) -> dict[str, Any]:
    T, D = a.shape
    heat = _signed(pool2d(a, 64, 128))
    norms = np.linalg.norm(a, axis=1)
    idx, coords, ratios = _pca2(a)
    return {
        "kind": "sequence",
        "shape": shape,
        "tokens": T,
        "features": D,
        "heatmap": _r(heat, 3),
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
