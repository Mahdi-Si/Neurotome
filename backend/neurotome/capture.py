"""Capture: run one forward pass through TorchLens and turn it into Neurotome's IR.

This module is the ONLY place that talks to TorchLens. Everything downstream
(graph views, tensor views, the API, the frontend) reads the IR below, so a
TorchLens upgrade or a different capture backend only has to change this file.

IR schema (ir.json)::

    {
      "model":  {"class_name", "num_params", "num_ops", "flops", "activation_bytes",
                 "truncated"        (an op was skipped for the budget), "budget_bytes" | null,
                 "capture_modules"  ([addresses] | null), "saved_ops" (non-io ops saved)}  (last 3: M4)
                 "backend" ("torchlens" | "nnsight"), "granularity" ("op" | "module")  (v0.2, absent before)
      "ops": [{
          "id":        stable id, e.g. "conv2d_3_8" (".p2" suffix for 2nd pass of a recurrent op)
          "type":      "conv2d" | "relu" | "iadd" | "input" | "output" | ...
          "func":      torch function name
          "shape":     [1, 64, 56, 56]
          "dtype":     "torch.float32"
          "device":    "cpu" | "cuda:0" | null  (device of the first output tensor; absent before M1)
          "parents":   [op ids]    "children": [op ids]
          "modules":   ["layer1:1", "layer1.0:1", "layer1.0.conv2:1"]  module-call stack, outermost first
          "module":    "layer1.0.conv2" or null (the leaf module that produced it)
          "num_params", "flops", "is_input", "is_output", "saved"
          "unsaved":   null (saved) | "budget" | "modules" | "no_tensor"   (absent before M4)
          "stats":     {"mean","std","min","max","zero_frac","hist":[32 ints],"hist_range":[lo,hi],"numel",
                        "nan","inf"}  moments/hist over finite values; nan/inf counts absent before M1
      }],
      "modules": {"layer1.0": {"class_name", "num_params", "depth"}, ...}
    }
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable
from typing import Any, cast

import numpy as np
import torch

from .errors import NO_GRAD, TraceError
from .graph import strip_call
from .interventions import diff_row
from .jobs import Progress
from .store import RunStore
from .tensors import safe_range


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - TorchLens properties can raise on exotic ops
        return default


def tensor_stats(a: np.ndarray, bins: int = 32) -> dict[str, Any]:
    flat = np.asarray(a, dtype=np.float32).ravel()
    finite = flat[np.isfinite(flat)]
    counts = {"nan": int(np.isnan(flat).sum()), "inf": int(np.isinf(flat).sum())}
    if finite.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "zero_frac": None,
            "hist": [0] * bins,
            "hist_range": [0.0, 1.0],
            "numel": int(flat.size),
            **counts,
        }
    lo, hi = float(finite.min()), float(finite.max())
    hist, _ = np.histogram(finite.astype(np.float64), bins=bins, range=safe_range(lo, hi))
    return {
        "mean": float(finite.mean(dtype=np.float64)),
        "std": float(finite.std(dtype=np.float64)),
        "min": lo,
        "max": hi,
        "zero_frac": float(np.count_nonzero(finite == 0) / finite.size),
        "hist": hist.astype(int).tolist(),
        "hist_range": [lo, hi],
        "numel": int(flat.size),
        **counts,
    }


def _run_trace(model: torch.nn.Module, inputs: Any, save_grads: bool = False, **kw):
    import torchlens as tl

    args = list(inputs) if isinstance(inputs, (tuple, list)) else inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, args, layers_to_save="all", save_grads=save_grads, unwrap_when_done=True, **kw)


def _visible_ops(trace) -> list:
    return [
        op
        for op in trace.layer_list
        if not _safe(lambda op=op: op.is_buffer, False)
        and (_safe(lambda op=op: op.is_compute_op, False) or op.is_input or op.is_output)
    ]


def _id_maps(ops: Iterable) -> tuple[dict[int, str], dict[str, str]]:
    """Map each TorchLens op to a stable id, and every label TorchLens may use to that id."""
    by_obj: dict[int, str] = {}
    by_label: dict[str, str] = {}
    for op in ops:
        passes = _safe(lambda op=op: op.num_passes, 1) or 1
        pid = op.layer_label if passes == 1 else f"{op.layer_label}.p{_safe(lambda op=op: op.pass_index, 1)}"
        by_obj[id(op)] = pid
        by_label[op.label] = pid
        if passes == 1:
            by_label[op.layer_label] = pid
    return by_obj, by_label


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for v in value:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _trace_or_raise(model: torch.nn.Module, inputs: Any, save_grads: bool = False, intervene=None):
    """``intervene``: TorchLens capture-time hooks; a ValueError they raise reaches the caller unchanged."""
    try:
        return _run_trace(model, inputs, save_grads, **({"intervene": intervene} if intervene else {}))
    except Exception as e:  # noqa: BLE001 - anything TorchLens or forward() raises
        if isinstance(e, torch.cuda.OutOfMemoryError) or (intervene and isinstance(e, ValueError)):
            raise
        first = (str(e).strip().splitlines() or [""])[0][:300]
        raise TraceError(
            f"TorchLens could not trace forward(): {type(e).__name__}: {first}. Check that model(x) runs on "
            "this input outside Neurotome; ops TorchLens cannot record (custom CUDA kernels, C++ extensions) "
            "are the usual cause."
        ) from e


def _module_table(trace) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for m in _safe(lambda: list(trace.modules), []) or []:
        addr = str(m.address)
        modules["" if addr == "self" else addr] = {
            "class_name": str(_safe(lambda m=m: m.class_name, "")),
            "num_params": int(_safe(lambda m=m: m.num_params, 0) or 0),
            "depth": int(_safe(lambda m=m: m.address_depth, 0) or 0),
        }
    return modules


def _check_modules(selection: list[str], table: dict[str, Any]) -> None:
    bad = [m for m in selection if m not in table or not m]
    if bad:
        top = [a for a in table if a and "." not in a][:5]
        raise ValueError(
            f"Unknown module(s) {', '.join(bad)}; pick addresses from the module tree, e.g. {', '.join(top)}"
        )


def _out(value: Any) -> torch.Tensor:
    """``_first_tensor`` where the caller knows there is one (typing only)."""
    return cast(torch.Tensor, _first_tensor(value))


def capture(
    model: torch.nn.Module,
    inputs: Any,
    store: RunStore,
    run_id: str,
    max_bytes: int | None = None,
    modules: list[str] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Trace ``model(inputs)``, save activations to the run store and return the IR.

    ``modules``: module addresses whose ops are saved (None: every op). ``max_bytes``: the disk budget; it is
    never crossed: after the first non-io op that does not fit, every later non-io op is skipped. Input and
    output ops are always saved. Stats are computed for every op with a tensor, saved or not."""
    model.eval()
    if progress:
        progress(0.12, "Trace")
    trace = _trace_or_raise(model, inputs)
    table = _module_table(trace)
    selection = set(modules or ())
    if modules:
        _check_modules(modules, table)
    ops = _visible_ops(trace)
    by_obj, by_label = _id_maps(ops)
    outs = [_first_tensor(_safe(lambda op=op: op.out)) for op in ops]
    # ponytail: memory during the trace is unchanged; pass TorchLens layer labels when big models need it
    total = sum(t.numel() * 4 for op, t in zip(ops, outs, strict=True) if t is not None and _io(op))
    over = False

    ir_ops: list[dict[str, Any]] = []
    disk_bytes = 0
    n, step = len(ops), max(1, len(ops) // 50)
    for i, (op, out) in enumerate(zip(ops, outs, strict=True), start=1):
        oid = by_obj[id(op)]
        unsaved, stats = None, None
        if out is None:
            unsaved = "no_tensor"
        elif not _io(op):
            need = out.numel() * 4
            stack = _safe(lambda op=op: op.modules, []) or []
            inside = not selection or any(strip_call(str(c)) in selection for c in stack)
            if not inside:
                unsaved = "modules"
            elif over or (max_bytes is not None and total + need > max_bytes):
                over, unsaved = True, "budget"
            else:
                total += need
        if out is not None:
            arr = out.detach().to("cpu", torch.float32).numpy()
            stats = tensor_stats(arr)
            if unsaved is None:
                disk_bytes += store.save_tensor(run_id, oid, arr)
            del arr
        shape = _safe(lambda op=op: list(op.shape), None) or (list(out.shape) if out is not None else [])
        ir_ops.append(
            {
                "id": oid,
                "type": "input" if op.is_input else "output" if op.is_output else str(op.layer_type),
                "func": _safe(lambda op=op: str(op.func_name), None),
                "shape": [int(s) for s in shape],
                "dtype": str(_safe(lambda op=op: op.dtype, "")),
                "device": str(out.device) if out is not None else None,
                "parents": [
                    by_label[p] for p in (_safe(lambda op=op: op.parents, []) or []) if p in by_label
                ],
                "children": [
                    by_label[c] for c in (_safe(lambda op=op: op.children, []) or []) if c in by_label
                ],
                "modules": [str(m) for m in (_safe(lambda op=op: op.modules, []) or [])],
                "module": _safe(lambda op=op: op.atomic_module_address, None),
                "num_params": int(_safe(lambda op=op: op.num_params, 0) or 0),
                "flops": int(_safe(lambda op=op: op.flops_forward, 0) or 0),
                "is_input": bool(op.is_input),
                "is_output": bool(op.is_output),
                "saved": unsaved is None,
                "unsaved": unsaved,
                "stats": stats,
            }
        )
        if progress and (i % step == 0 or i == n):
            progress(0.2 + 0.7 * i / n, f"Save tensors {i}/{n}")

    body = [o for o in ir_ops if not (o["is_input"] or o["is_output"])]
    ir = {
        "model": {
            "class_name": type(model).__name__,
            "num_params": int(sum(p.numel() for p in model.parameters())),
            "num_ops": len(body),
            "flops": int(_safe(lambda: trace.total_flops_forward, 0) or 0),
            "activation_bytes": disk_bytes,
            "truncated": over,
            "budget_bytes": max_bytes,
            "capture_modules": list(modules) if modules else None,
            "saved_ops": sum(1 for o in body if o["saved"]),
            "backend": "torchlens",
            "granularity": "op",
        },
        "ops": ir_ops,
        "modules": table,
    }
    return ir


def _io(op) -> bool:
    return bool(op.is_input or op.is_output)


def _grad_leaf(x: Any) -> Any:
    """Float inputs require grad, so node gradients also reach frozen or parameter-free models."""
    if torch.is_tensor(x) and x.is_floating_point():
        return x.detach().clone().requires_grad_(True)
    return type(x)(_grad_leaf(i) for i in x) if isinstance(x, (tuple, list)) else x


def op_gradients(
    model: torch.nn.Module, inputs: Any, target: int | None, to_logits, objective: dict | None = None
) -> dict[str, Any]:
    """Re-trace with gradients and return per-op gradient × activation for one objective.

    ``objective`` None: back-propagate ``logits[0, target]`` (target None -> argmax).
    ``objective`` ``{"op": op_id, "channel": c}``: back-propagate ``first_tensor(op.out)[0, c].sum()``;
    ops downstream of the unit get no gradient and so no score.

    Returns ``{"target", "logit", "value", "scores": {op_id: signed sum(act*grad)},
    "channel_scores": {op_id: [per-channel sums]}, "merges": {child_op: {parent_op: float}}}``.
    ``value`` is the back-propagated scalar; ``target`` / ``logit`` are None for a unit objective.
    ``merges`` holds, for every op with >= 2 visible parents, each parent's grad × input *through that op*.
    For linear merges (add, sub, cat, stack) the contributions of an op sum to its own grad × act score.
    """
    model.eval()
    trace = _trace_or_raise(model, _grad_leaf(inputs), save_grads=True)
    ops = _visible_ops(trace)
    by_obj, _ = _id_maps(ops)
    logit = None
    if objective is None:
        out_op = next(op for op in ops if op.is_output)
        logits = to_logits(_first_tensor(out_op.out))
        if target is None:
            target = int(logits[0].detach().argmax())
        root = logits[0, target]
        logit = float(root.detach())
    else:
        op_id, c = objective["op"], int(objective["channel"])
        unit = next((op for op in ops if by_obj[id(op)] == op_id), None)
        out = _first_tensor(_safe(lambda: unit.out)) if unit is not None else None
        if out is None:
            raise ValueError(f"Unknown op {op_id}")
        if out.dim() < 2 or not 0 <= c < out.shape[1]:
            raise ValueError(f"Channel {c} out of range for {op_id} with shape {list(out.shape)}")
        target, root = None, out[0, c].sum()
    if not root.requires_grad:
        raise ValueError(NO_GRAD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.log_backward(root)

    scores: dict[str, float] = {}
    channel_scores: dict[str, list[float]] = {}
    for op in ops:
        act = _first_tensor(_safe(lambda op=op: op.out))
        grad = _first_tensor(_safe(lambda op=op: op.grad))
        if act is None or grad is None or act.shape != grad.shape or not act.is_floating_point():
            continue
        gxa = (act.detach() * grad.detach()).to(torch.float32)
        oid = by_obj[id(op)]
        scores[oid] = float(gxa.sum())
        if gxa.dim() >= 3:  # (B, C, ...) -> per channel
            channel_scores[oid] = gxa[0].flatten(1).sum(1).tolist()
    return {
        "target": target,
        "logit": logit,
        "value": float(root.detach()),
        "scores": scores,
        "channel_scores": channel_scores,
        "merges": _merges(trace, ops, by_obj),
    }


def _merges(trace, ops: list, by_obj: dict[int, str]) -> dict[str, dict[str, float]]:
    every: dict[str, Any] = {}  # TorchLens label -> op, buffers included
    for op in trace.layer_list:
        every[op.label] = op
        if (_safe(lambda op=op: op.num_passes, 1) or 1) == 1:
            every[op.layer_label] = op
    out = {}
    for op in ops:
        try:
            m = _merge(op, every, by_obj)
        except Exception:  # noqa: BLE001 - an op that cannot be re-called falls back to parent shares
            continue
        if m:
            out[by_obj[id(op)]] = m
    return out


def _merge(op, every: dict[str, Any], by_obj: dict[int, str]) -> dict[str, float] | None:
    """Each visible parent's grad × input through ``op``: re-call op.func on cloned parent leaves."""
    grad = _first_tensor(_safe(lambda: op.grad))
    parents = {k: every[label] for k, label in op.parent_arg_positions["args"].items()}
    if grad is None or len({by_obj[id(p)] for p in parents.values() if id(p) in by_obj}) < 2:
        return None
    leaves: dict[str, torch.Tensor] = {}
    slots: dict[int, Any] = {}
    for k, p in parents.items():
        t = _out(p.out).detach()
        if id(p) in by_obj:  # visible parent -> one leaf per op; buffers stay constants
            t = leaves.setdefault(by_obj[id(p)], t.clone().requires_grad_(True)).clone()  # in-place funcs
        if isinstance(k, tuple):  # (i, j): element j of a list at position i, e.g. cat([a, b])
            slots.setdefault(k[0], {})[k[1]] = t
        else:
            slots[k] = t
    rest = iter(op.non_tensor_pos_args)
    args = [
        ([v for _, v in sorted(slots[i].items())] if isinstance(slots[i], dict) else slots[i])
        if i in slots
        else next(rest)  # params are not recorded here: too few args -> StopIteration -> skipped
        for i in range(op.num_pos_args)
    ]
    if next(rest, rest) is not rest:
        return None
    with torch.enable_grad():
        y = _out(op.func(*args, **op.non_tensor_kwargs))
        grads = torch.autograd.grad(y, list(leaves.values()), grad.to(y.dtype), allow_unused=True)
    return {
        pid: 0.0 if g is None else float((g.detach() * leaf.detach()).float().sum())
        for (pid, leaf), g in zip(leaves.items(), grads, strict=True)
    }


@torch.enable_grad()  # as at capture: no_grad sends nn.MultiheadAttention down its fused path (other op ids)
def intervene(
    model: torch.nn.Module,
    inputs: Any,
    edits: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    downstream: Collection[str],
    keep: Collection[str] = (),
) -> dict[str, Any]:
    """One traced forward pass with every edit applied where its op runs (TorchLens capture-time hooks; edits
    on one op compose in list order). Returns ``{"nodes": {op: diff row}, "stats": {op: tensor_stats(after)},
    "after": {op: float32 array for op in keep}, "logits_after": the output op's tensor}``; nodes / stats
    cover the downstream ops with a float tensor. TorchLens DAG replay stops at multi-pass ops (GPT-2's
    residual adds), so this re-traces instead of replaying."""
    # ponytail: two full traces; store raw_index in the IR at capture to skip the clean one for big models
    model.eval()
    clean = _visible_ops(_trace_or_raise(model, inputs))
    by_obj, _ = _id_maps(clean)
    ids = [by_obj[id(o)] for o in clean]
    raw = {by_obj[id(o)]: o.raw_index for o in clean}
    down = set(downstream)
    before = {oid: _numpy(o) for o in clean if (oid := by_obj[id(o)]) in down}
    fns: dict[str, list] = {}
    for op, fn in edits:
        if op not in raw:
            raise ValueError(f"Unknown op {op}")
        fns.setdefault(op, []).append(fn)
    returned: dict[str, torch.Tensor] = {}

    def make(op: str):
        def hook(out, *, hook):  # TorchLens passes the hook spec keyword-only
            if not torch.is_tensor(out):
                raise ValueError(f"{op} returns several tensors; edit one of its consumers")
            new = out
            for fn in fns[op]:
                new = fn(new)
            if tuple(new.shape) != tuple(out.shape):
                raise ValueError(f"Edit of {op} changed its shape {list(out.shape)} -> {list(new.shape)}")
            returned[op] = new.detach().clone()  # a later in-place op may change the live tensor
            return new

        return hook

    hooks = {raw[op]: make(op) for op in fns}
    edited = _visible_ops(_trace_or_raise(model, inputs, intervene=lambda ctx: hooks.get(ctx.raw_index)))
    by_obj2, by_label = _id_maps(edited)
    if [by_obj2[id(o)] for o in edited] != ids:
        raise ValueError("The intervention changed the model's control flow; TorchLens cannot align the ops")
    ops = {by_obj2[id(o)]: o for o in edited}
    for op in fns:
        rec = _first_tensor(_safe(lambda op=op: ops[op].out))
        if op not in returned or rec is None or not _same(rec.detach(), returned[op]):
            raise ValueError(_cannot_edit(op, ops[op], by_label))
    nodes, stats, after = {}, {}, {}
    for op, b in before.items():
        a = _numpy(ops[op])
        if a is None or b is None or a.shape != b.shape:
            continue
        nodes[op], stats[op] = diff_row(b, a), tensor_stats(a)
        if op in keep:
            after[op] = a
    out_op = next(o for o in edited if o.is_output)
    return {"nodes": nodes, "stats": stats, "after": after, "logits_after": _out(out_op.out).detach()}


def _numpy(op) -> np.ndarray | None:
    """A float32 CPU copy of a float op's first tensor, else None."""
    t = _first_tensor(_safe(lambda: op.out))
    if t is None or not t.is_floating_point():
        return None
    return t.detach().to("cpu", torch.float32).numpy().copy()


def _cannot_edit(op_id: str, op, by_label: dict[str, str]) -> str:
    """An op whose edit TorchLens did not keep (it returns its input object, e.g. eval-mode dropout)."""
    parent = next((by_label[p] for p in (_safe(lambda: op.parents, []) or []) if p in by_label), "?")
    return (
        f"TorchLens cannot edit {op_id} ({_safe(lambda: op.func_name, op.layer_type)}): it returns its input "
        f"unchanged in eval mode. Edit its parent {parent} instead."
    )


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(((a == b) | (a.isnan() & b.isnan())).all())
