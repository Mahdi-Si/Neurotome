"""Interventions: the edit set (normalise, hash, labels), edit functions and greedy generation.

Plain torch / numpy: no TorchLens, no NNsight. The capture modules decide *where* an edit runs
(``capture.intervene``, ``capture_nnsight.intervene``); this module decides *what* it computes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any

import numpy as np
import torch

from .models import to_logits

KINDS = ("zero", "mean", "resample", "constant", "scale", "steer")
MAX_EDITS = 32
MAX_NEW_TOKENS = 64
GEN_KINDS = ("zero", "constant", "scale", "steer")  # + mean over positions
CH_AXIS = {4: 1, 3: 2, 2: 1}
SCOPED = "(B, C, H, W), (B, T, D) or (B, D)"

Fn = Callable[[torch.Tensor], torch.Tensor]


def normalise(raw: dict[str, Any], ir: dict[str, Any]) -> dict[str, Any]:
    """The canonical InterventionSet (spec §1.3); ValueError with a user-facing sentence otherwise."""
    edits = raw.get("edits") or []
    if not edits:
        raise ValueError("An intervention set needs at least one edit")
    if len(edits) > MAX_EDITS:
        raise ValueError(f"At most {MAX_EDITS} edits per set")
    ops = {o["id"]: o for o in ir["ops"]}
    target = raw.get("target")
    return {"target": None if target is None else int(target), "edits": [_edit(dict(e), ops) for e in edits]}


def _edit(e: dict[str, Any], ops: dict[str, dict[str, Any]]) -> dict[str, Any]:
    op, kind = str(e.get("op")), e.get("kind")
    if kind not in KINDS:
        raise ValueError(f"Edit kind must be one of {', '.join(KINDS)}")
    o = ops.get(op)
    if o is None:
        raise ValueError(f"Unknown op {op}")
    if o["is_input"] or o["is_output"]:
        raise ValueError(f"{op} is the model's input or output; pick an op inside the model")
    dtype = o.get("dtype") or ""
    if "float" not in dtype:
        raise ValueError(f"{op} holds {dtype or 'no'} values; only float tensors can be edited")
    shape = list(o["shape"])
    out: dict[str, Any] = {"op": op, "kind": kind, **_scope(op, shape, e.get("channels"), e.get("positions"))}
    if kind == "mean":
        out["over"] = e.get("over") or "positions"
    elif kind == "resample":
        src = e.get("source_run")
        if not src:
            raise ValueError('Resample needs source_run (a run id or "random")')
        out["source_run"] = src
        if src == "random":
            out["seed"] = int(e.get("seed") or 0)
    elif kind == "constant":
        out["value"] = float(e.get("value") or 0.0)
    elif kind in ("scale", "steer"):
        out["alpha"] = float(1.0 if e.get("alpha") is None else e["alpha"])
    if kind == "steer":
        out["direction"] = _direction(op, shape, e.get("direction"))
    return out


def _scope(op: str, shape: list[int], channels: Any, positions: Any) -> dict[str, Any]:
    rank = len(shape)
    channels = sorted({int(c) for c in channels}) if channels else None
    if positions is not None and not isinstance(positions, str) and not len(positions):
        positions = None
    if (channels or positions is not None) and rank not in CH_AXIS:
        raise ValueError(f"{op} has shape {shape}: channel / position scopes need a {SCOPED} tensor")
    if channels:
        n = shape[CH_AXIS[rank]]
        bad = next((c for c in channels if not 0 <= c < n), None)
        if bad is not None:
            raise ValueError(f"Channel {bad} out of range for {op} ({n} channels)")
    if positions is None:
        return {"channels": channels, "positions": None}
    if rank == 2:
        raise ValueError(f"{op} has no positions (shape {shape}); scope it by channels")
    if positions == "last" or isinstance(positions, str):
        if rank != 3 or positions != "last":
            raise ValueError('"last" applies to token positions (B, T, D)')
        return {"channels": channels, "positions": "last"}
    if rank == 4:
        H, W = shape[2], shape[3]
        ok = all(isinstance(p, (list, tuple)) and len(p) == 2 for p in positions)
        pairs = sorted({(int(p[0]), int(p[1])) for p in positions}) if ok else []
        if not ok or any(not (0 <= y < H and 0 <= x < W) for y, x in pairs):
            raise ValueError(f"Positions of {op} are [y, x] pixels (H {H}, W {W})")
        return {"channels": channels, "positions": [list(p) for p in pairs]}
    T = shape[1]
    ok = all(isinstance(p, (int, np.integer)) and not isinstance(p, bool) for p in positions)
    if not ok or any(not 0 <= p < T for p in positions):
        raise ValueError(f"Positions of {op} are token indices below {T}")
    return {"channels": channels, "positions": sorted({int(p) for p in positions})}


def _direction(op: str, shape: list[int], d: Any) -> dict[str, Any]:
    d = dict(d or {})
    kind = d.get("kind")
    need = {"contrast": ("run_a", "run_b"), "channel": ("channel",)}.get(kind)
    if need is None or any(d.get(k) in (None, "") for k in need):
        raise ValueError("Steering needs a direction")
    if len(shape) not in CH_AXIS:
        raise ValueError(f"Steering needs a {SCOPED} tensor; {op} is {shape}")
    if kind == "channel":
        c, n = int(d["channel"]), shape[CH_AXIS[len(shape)]]
        if not 0 <= c < n:
            raise ValueError(f"Direction channel {c} out of range for {op} ({n} channels)")
        return {"kind": "channel", "channel": c}
    pool = d.get("pool") or "mean"
    pool = pool if len(shape) == 3 else "mean"  # "last" pools tokens; other ranks pool by mean
    return {"kind": "contrast", "run_a": d["run_a"], "run_b": d["run_b"], "pool": pool}


def set_hash(norm: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(norm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def downstream(ir: dict[str, Any], ops: Iterable[str]) -> list[str]:
    """The ops and every IR descendant, in IR order."""
    children = {o["id"]: o["children"] for o in ir["ops"]}
    seen, todo = set(), list(ops)
    while todo:
        if (op := todo.pop()) not in seen:
            seen.add(op)
            todo.extend(children.get(op, []))
    return [o["id"] for o in ir["ops"] if o["id"] in seen]


def keep_ops(ir: dict[str, Any], down: list[str], budget: int | None) -> tuple[list[str], bool]:
    """(saved, truncated): the downstream ops the clean run saved as float tensors, in IR order, under
    ``budget`` bytes (numel × 4) with the never-cross rule."""
    ops = {o["id"]: o for o in ir["ops"]}
    saved, total = [], 0
    for op in down:
        o = ops[op]
        if not o.get("saved") or "float" not in (o.get("dtype") or ""):
            continue
        need = math.prod(o["shape"]) * 4
        if budget is not None and total + need > budget:
            return saved, True
        saved.append(op)
        total += need
    return saved, False


def scope_mask(shape: Any, channels: list[int] | None, positions: Any) -> torch.Tensor | None:
    """Bool mask broadcastable to ``shape`` (None = everything); ``"last"`` is T − 1 of this shape."""
    rank = len(shape)
    mask = None
    if channels:
        ax = CH_AXIS[rank]
        m = torch.zeros(shape[ax], dtype=torch.bool)
        m[list(channels)] = True
        mask = m.view([-1 if i == ax else 1 for i in range(rank)])
    if positions is not None:
        if rank == 4:
            p = torch.zeros(shape[2], shape[3], dtype=torch.bool)
            ys, xs = zip(*positions, strict=True)
            p[list(ys), list(xs)] = True
            p = p.view(1, 1, shape[2], shape[3])
        else:
            p = torch.zeros(shape[1], dtype=torch.bool)
            p[[shape[1] - 1] if positions == "last" else list(positions)] = True
            p = p.view(1, -1, 1)
        mask = p if mask is None else mask & p
    return mask


def _mean_positions(t: torch.Tensor) -> torch.Tensor:
    dims = (1,) if t.dim() in (2, 3) else tuple(range(2, t.dim())) or (t.dim() - 1,)
    return t.mean(dim=dims, keepdim=True).expand_as(t).clone()  # own memory: the model may edit it in place


def edit_fn(edit: dict[str, Any], value: np.ndarray | None) -> Fn:
    """``f(x) = where(mask, r(x), x)`` for one normalised edit. ``value``: the resample / runs-mean
    replacement or the steer direction v (a channel vector), cast to x's dtype and device on first use."""
    kind, cache = edit["kind"], {}

    def val(t: torch.Tensor) -> torch.Tensor:
        key = (t.dtype, t.device)
        if key not in cache:
            cache[key] = torch.as_tensor(value, dtype=t.dtype, device=t.device)
        return cache[key]

    def replace(t: torch.Tensor) -> torch.Tensor:
        if kind == "zero":
            return torch.zeros_like(t)
        if kind == "mean" and edit["over"] == "positions":
            return _mean_positions(t)
        if kind in ("mean", "resample"):
            return val(t).expand_as(t).clone()  # never hand out the cached value (in-place ops downstream)
        if kind == "constant":
            return torch.full_like(t, edit["value"])
        if kind == "scale":
            return t * edit["alpha"]
        ax, d = CH_AXIS[t.dim()], edit["direction"]
        if d["kind"] == "channel":
            v = torch.zeros(t.shape[ax], dtype=t.dtype, device=t.device)
            v[d["channel"]] = 1
        else:
            v = val(t)
        return t + edit["alpha"] * v.view([-1 if i == ax else 1 for i in range(t.dim())])

    def fn(t: torch.Tensor) -> torch.Tensor:
        mask = scope_mask(t.shape, edit.get("channels"), edit.get("positions"))
        r = replace(t)
        return r if mask is None else torch.where(mask.to(t.device), r, t)

    return fn


# ---- labels (spec §1.5) ------------------------------------------------------------------------------


def num(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def index_list(xs: list[int]) -> str:
    """[0, 3, 7, 8, 9] -> "0,3,7–9" (runs of 3 or more as a–b)."""
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out += [f"{xs[i]}–{xs[j]}"] if j - i >= 2 else [str(x) for x in xs[i : j + 1]]
        i = j + 1
    return ",".join(out)


def _scope_text(edit: dict[str, Any]) -> str:
    parts = []
    if edit.get("channels"):
        parts.append(f"ch {index_list(edit['channels'])}")
    pos = edit.get("positions")
    if pos == "last":
        parts.append("last token")
    elif pos and isinstance(pos[0], list):
        more = f" +{len(pos) - 4}" if len(pos) > 4 else ""
        parts.append("px " + " ".join(f"({y},{x})" for y, x in pos[:4]) + more)
    elif pos:
        parts.append(f"tokens {index_list(pos)}")
    return " · ".join(parts) or "all"


def label(edit: dict[str, Any], path: str, resolved: str | None = None, n_runs: int | None = None) -> str:
    kind = edit["kind"]
    if kind == "zero":
        name = "Zero"
    elif kind == "mean":
        name = "Mean over positions" if edit["over"] == "positions" else f"Mean over {n_runs} runs"
    elif kind == "resample" and edit["source_run"] == "random":
        name = f"Resample ← {resolved} (random · seed {edit['seed']})"
    elif kind == "resample":
        name = f"Replace ← {edit['source_run']}"
    elif kind == "constant":
        name = f"Constant {num(edit['value'])}"
    elif kind == "scale":
        name = f"Scale ×{num(edit['alpha'])}"
    else:
        a, d = edit["alpha"], edit["direction"]
        v = f"e_{d['channel']}" if d["kind"] == "channel" else f"({d['run_b']} − {d['run_a']}, {d['pool']})"
        name = f"Steer {'+' if a >= 0 else '-'}{num(abs(a))} × {v}"
    return f"{name} · {path} · {_scope_text(edit)}"


def diff_row(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """‖after − before‖₂, relative to ‖before‖, cosine and max |Δ| over positions where both are finite;
    ``nonfinite``: positions where either side is non-finite and the two differ (NaN = NaN, inf = inf)."""
    b = np.asarray(before, dtype=np.float64).ravel()
    a = np.asarray(after, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    nonfinite = int((~ok & ~same).sum())
    a, b = a[ok], b[ok]
    d = a - b
    l2, nb, na = float(np.linalg.norm(d)), float(np.linalg.norm(b)), float(np.linalg.norm(a))
    return {
        "l2": l2,
        "rel": l2 / nb if nb else 0.0,
        "cos": float(a @ b / (na * nb)) if na and nb else None,
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "nonfinite": nonfinite,
    }


# ---- generation (forward hooks on module outputs, no TorchLens) --------------------------------------


def _apply(output: Any, fn: Fn) -> Any:
    """``fn`` on a module output: the tensor itself, or the first tensor of a tuple / list / dict / HF
    ModelOutput (the rest kept). ValueError when the output holds no tensor: an edit never passes silently."""
    if torch.is_tensor(output):
        return fn(output)
    if isinstance(output, MutableMapping):  # ModelOutput is an OrderedDict and supports item assignment
        key = next((k for k, v in output.items() if torch.is_tensor(v)), None)
        if key is not None:
            output[key] = fn(output[key])
            return output
    elif isinstance(output, (tuple, list)):
        i = next((k for k, v in enumerate(output) if torch.is_tensor(v)), None)
        if i is not None:
            items = list(output)
            items[i] = fn(items[i])
            return type(output)(*items) if hasattr(output, "_fields") else type(output)(items)
    raise ValueError(f"Cannot edit a module output of type {type(output).__name__}: it holds no tensor")


def _hooked(model: torch.nn.Module, hooks: list[tuple[tuple[str, int], Fn]]) -> list:
    """Register forward hooks that apply each fn at call k of its module (counted per forward pass)."""
    sites: dict[str, dict[int, list[Fn]]] = {}
    for (address, call), fn in hooks:
        sites.setdefault(address, {}).setdefault(call, []).append(fn)
    counts: dict[str, int] = {}

    def post(address: str, _m, _args, output):
        counts[address] = k = counts.get(address, 0) + 1
        for fn in sites[address].get(k, []):
            output = _apply(output, fn)
        return output

    handles = [model.register_forward_pre_hook(lambda *_: counts.clear())]
    for address in sites:
        mod = model.get_submodule(address)
        handles.append(mod.register_forward_hook(lambda m, a, o, address=address: post(address, m, a, o)))
    return handles


def _greedy(model, ids: torch.Tensor, n: int, tokenizer, report, msg: str, base: float) -> dict[str, Any]:
    new: list[int] = []
    top: list[list[dict[str, Any]]] = []
    x = ids
    for i in range(n):
        report(base + 0.49 * i / n, f"{msg} {i + 1}/{n}")
        with torch.no_grad():
            p = torch.softmax(to_logits(model(x))[0].float(), -1)
        vals, idx = p.topk(min(5, p.numel()))
        top.append(
            [
                {
                    "id": int(j),
                    "token": tokenizer.decode([int(j)]),
                    "prob": round(v, 4) if math.isfinite(v) else None,
                }
                for v, j in zip(vals.tolist(), idx.tolist(), strict=True)
            ]
        )
        nxt = int(p.argmax())
        new.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=x.dtype, device=x.device)], 1)
    return {
        "ids": new,
        "tokens": [tokenizer.decode([i]) for i in new],
        "text": tokenizer.decode(new),
        "top": top,
    }


def generate(model, ids: torch.Tensor, hooks: list[tuple[tuple[str, int], Fn]], n: int, tokenizer,
             progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """Greedy decoding of ``n`` tokens after ``ids`` (1, T), once clean and once with ``hooks`` (no KV cache:
    every step re-runs the whole sequence, so ``"last"`` follows it). Returns clean / steered / diverge."""
    report = progress or (lambda _f, _m: None)
    clean = _greedy(model, ids, n, tokenizer, report, "Clean", 0.0)
    handles = _hooked(model, hooks)
    try:
        steered = _greedy(model, ids, n, tokenizer, report, "Intervened", 0.5)
    finally:
        for h in handles:
            h.remove()
    diverge = next(
        (i for i, (a, b) in enumerate(zip(clean["ids"], steered["ids"], strict=True)) if a != b), None
    )
    return {"clean": clean, "steered": steered, "diverge": diverge}
