"""Capture: run one forward pass through TorchLens and turn it into Neurotome's IR.

This module is the ONLY place that talks to TorchLens. Everything downstream
(graph views, tensor views, the API, the frontend) reads the IR below, so a
TorchLens upgrade or a different capture backend only has to change this file.

IR schema (ir.json)::

    {
      "model":  {"class_name", "num_params", "num_ops", "flops", "activation_bytes"},
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
          "stats":     {"mean","std","min","max","zero_frac","hist":[32 ints],"hist_range":[lo,hi],"numel",
                        "nan","inf"}  moments/hist over finite values; nan/inf counts absent before M1
      }],
      "modules": {"layer1.0": {"class_name", "num_params", "depth"}, ...}
    }
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

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
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": lo,
        "max": hi,
        "zero_frac": float(np.count_nonzero(finite == 0) / finite.size),
        "hist": hist.astype(int).tolist(),
        "hist_range": [lo, hi],
        "numel": int(flat.size),
        **counts,
    }


def _run_trace(model: torch.nn.Module, inputs: Any, save_grads: bool = False):
    import torchlens as tl

    args = list(inputs) if isinstance(inputs, (tuple, list)) else inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, args, layers_to_save="all", save_grads=save_grads, unwrap_when_done=True)


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


def capture(
    model: torch.nn.Module,
    inputs: Any,
    store: RunStore,
    run_id: str,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Trace ``model(inputs)``, save every activation to the run store and return the IR."""
    model.eval()
    trace = _run_trace(model, inputs)
    ops = _visible_ops(trace)
    by_obj, by_label = _id_maps(ops)

    ir_ops: list[dict[str, Any]] = []
    total_bytes = 0
    for op in ops:
        oid = by_obj[id(op)]
        out = _first_tensor(_safe(lambda op=op: op.out))
        saved = False
        stats = None
        if out is not None and (max_bytes is None or total_bytes < max_bytes):
            arr = out.detach().to("cpu", torch.float32).numpy()
            stats = tensor_stats(arr)
            total_bytes += store.save_tensor(run_id, oid, arr)
            saved = True
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
                "saved": saved,
                "stats": stats,
            }
        )

    modules: dict[str, dict[str, Any]] = {}
    for m in _safe(lambda: list(trace.modules), []) or []:
        addr = str(m.address)
        modules["" if addr == "self" else addr] = {
            "class_name": str(_safe(lambda m=m: m.class_name, "")),
            "num_params": int(_safe(lambda m=m: m.num_params, 0) or 0),
            "depth": int(_safe(lambda m=m: m.address_depth, 0) or 0),
        }

    ir = {
        "model": {
            "class_name": type(model).__name__,
            "num_params": int(sum(p.numel() for p in model.parameters())),
            "num_ops": sum(1 for o in ir_ops if not (o["is_input"] or o["is_output"])),
            "flops": int(_safe(lambda: trace.total_flops_forward, 0) or 0),
            "activation_bytes": total_bytes,
            "truncated": max_bytes is not None and total_bytes >= max_bytes,
        },
        "ops": ir_ops,
        "modules": modules,
    }
    return ir


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
    trace = _run_trace(model, inputs, save_grads=True)
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
        t = _first_tensor(p.out).detach()
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
        y = _first_tensor(op.func(*args, **op.non_tensor_kwargs))
        grads = torch.autograd.grad(y, list(leaves.values()), grad.to(y.dtype), allow_unused=True)
    return {
        pid: 0.0 if g is None else float((g.detach() * leaf.detach()).float().sum())
        for (pid, leaf), g in zip(leaves.items(), grads, strict=True)
    }


def patch(
    model: torch.nn.Module,
    inputs: Any,
    op_id: str,
    replace: Callable[[torch.Tensor], torch.Tensor],
    keep: Iterable[str] = (),
) -> dict[str, Any]:
    """Re-run the forward pass with op_id's output replaced (TorchLens DAG replay).

    Returns ``{"deltas": {op: ||after - before||_2},
    "rel": {op: ||delta|| / ||before|| (0 when ||before|| = 0)},
    "after": {op: tensor for op in keep}, "logits_after": first tensor of the output op}``.
    """
    import torchlens as tl

    model.eval()
    args = list(inputs) if isinstance(inputs, (tuple, list)) else inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, args, layers_to_save="all", intervention_ready=True, unwrap_when_done=True)
    ops = _visible_ops(trace)
    by_obj, by_label = _id_maps(ops)
    before: dict[str, torch.Tensor] = {}
    for op in ops:
        t = _first_tensor(_safe(lambda op=op: op.out))
        if t is not None:
            before[by_obj[id(op)]] = t.detach().clone()  # before the replay mutates anything
    op = next((o for o in ops if by_obj[id(o)] == op_id), None)
    if op is None or op_id not in before:
        raise ValueError(f"Unknown op {op_id}")
    new = replace(before[op_id])
    if tuple(new.shape) != tuple(before[op_id].shape):
        raise ValueError(f"replacement shape {list(new.shape)} != {list(before[op_id].shape)}")
    label = op.layer_label if op_id == op.layer_label else op.label  # .pN ids -> pass-qualified label
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.do(tl.label(label), tl.replace_with(new), engine="replay", confirm_mutation=True)

    # Replay re-runs in-place funcs (x += y, ReLU(inplace)) on their parents' tensors, so an op whose tensor a
    # later in-place op mutates reads that op's value. Ops outside the patched op's descendants cannot change
    # (delta 0); a mutated descendant's own value is lost, so it is left out.
    down, todo = {op_id}, [op]
    while todo:
        for c in _safe(lambda o=todo.pop(): o.children, []) or []:
            if (cid := by_label.get(c)) and cid not in down:
                down.add(cid)
                todo.append(next(o for o in ops if by_obj[id(o)] == cid))
    outs = [(by_obj[id(o)], _first_tensor(_safe(lambda o=o: o.out))) for o in ops]
    lost, mutated = set(), set()
    for o, (oid, t) in zip(reversed(ops), reversed(outs), strict=True):
        if t is not None:
            if t.data_ptr() in mutated:
                lost.add(oid)
            if _safe(lambda o=o: o.is_inplace, False):
                mutated.add(t.data_ptr())

    keep = set(keep)
    deltas: dict[str, float] = {}
    rel: dict[str, float] = {}
    after: dict[str, torch.Tensor] = {}
    for oid, a in outs:
        b = before.get(oid)
        if a is None or b is None or a.shape != b.shape or not b.is_floating_point():
            continue
        if oid not in down:
            a = b
        elif oid in lost:
            continue
        a = a.detach()
        d, n = float((a.float() - b.float()).norm()), float(b.float().norm())
        deltas[oid], rel[oid] = d, (d / n if n else 0.0)
        if oid in keep:
            after[oid] = a.clone()
    out_op = next(o for o in ops if o.is_output)
    return {"deltas": deltas, "rel": rel, "after": after, "logits_after": _first_tensor(out_op.out).detach()}
