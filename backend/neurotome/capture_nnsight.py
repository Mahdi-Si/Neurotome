"""Capture with NNsight: module outputs only, for models too big to record op by op.

This module is the ONLY place that talks to NNsight. It writes the same IR as capture.py (see its docstring),
at module granularity: one op per leaf-module call, id ``f"{address}@{call}"`` (``block.relu@2``), plus
``input_<k>`` and ``output_1``. NNsight runs the forward; plain forward (pre-)hooks give call order, the
module-call stack, tensor pointers (edges by tensor identity) and the values, copied at the hook because a
later in-place op (``ReLU(inplace=True)``, ``+=``) changes the live tensor. Functional ops between modules
(residual adds, softmax outside a module) are invisible; an input no module produced falls back to the
previous leaf call as its parent. ``flops`` is 0. Interventions edit leaf-module calls natively
(``env.output = fn(env.output)``).
"""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Any

import torch

from .capture import _check_modules, _first_tensor, _grad_leaf, tensor_stats
from .errors import NO_GRAD, TraceError
from .interventions import diff_row
from .jobs import Progress
from .store import RunStore

MISSING = "The nnsight backend needs the nnsight extra: pip install 'neurotome[nnsight]'"


def available() -> bool:
    return importlib.util.find_spec("nnsight") is not None


@dataclass
class _Call:
    address: str
    call: int
    modules: list[str]  # call stack "addr:call", outermost first, root excluded, own call last
    inputs: list[int]  # data_ptrs of the tensors in args / kwargs (one level into tuples / lists)
    out: torch.Tensor | None
    value: Any = None  # capture: float32 numpy copy (kept modules only); gradients: graph-attached clone
    stats: dict | None = None
    grad: torch.Tensor | None = None  # gradients: w.r.t. the value before any later in-place op

    @property
    def id(self) -> str:
        return f"{self.address}@{self.call}"


@dataclass
class _Recorder:
    """Forward (pre-)hooks on every module for one forward pass: leaf calls in execution order."""

    model: torch.nn.Module
    grads: bool = False  # op_gradients: clone every leaf output and hook its gradient
    keep: set[str] = field(default_factory=set)  # capture: leaf addresses whose values are kept
    calls: list[_Call] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    output: Any = None

    def __enter__(self) -> _Recorder:
        self._stack: list[tuple[str, int, list[int]]] = []
        self._leaves = {a for a, m in self.model.named_modules() if next(m.children(), None) is None}
        self._handles = []
        for a, m in self.model.named_modules():
            self._handles.append(m.register_forward_pre_hook(partial(self._pre, a), with_kwargs=True))
            self._handles.append(m.register_forward_hook(partial(self._post, a)))
        return self

    def __exit__(self, *_exc) -> None:
        for h in self._handles:
            h.remove()

    def _pre(self, address: str, _mod, args, kwargs) -> None:
        self.counts[address] = n = self.counts.get(address, 0) + 1
        self._stack.append((address, n, _ptrs((*args, *kwargs.values()))))

    def _post(self, address: str, _mod, _args, output) -> None:
        if address in self._leaves:
            out = _first_tensor(output)
            stack = [f"{a}:{n}" for a, n, _ in self._stack if a]
            # ponytail: every leaf output stays referenced until the trace ends so data_ptrs cannot be reused
            # (wrong edges); a storage-lifetime check would free them early when GPU memory binds
            call = _Call(address, self._stack[-1][1], stack, self._stack[-1][2], out)
            if out is not None and self.grads:
                call.value = out.clone()
                if out.requires_grad:
                    out.register_hook(partial(setattr, call, "grad"))
            elif out is not None:
                # ponytail: kept values wait in RAM as float32 until saved; stream them if models outgrow it
                arr = out.detach().to("cpu", torch.float32, copy=True).numpy()
                call.stats = tensor_stats(arr)
                call.value = arr if address in self.keep else None
            self.calls.append(call)
        if not address:
            self.output = output
        self._stack.pop()


def _ptrs(values) -> list[int]:
    out = []
    for v in values:
        for t in v if isinstance(v, (tuple, list)) else (v,):
            if isinstance(t, torch.Tensor):
                out.append(t.data_ptr())
    return out


def _args(inputs: Any) -> tuple:
    return tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)


def _output_tensor(value: Any) -> torch.Tensor | None:
    """The tensor a model output stands for: ``.logits`` when present, else the first tensor."""
    logits = getattr(value, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    return _first_tensor(list(value.values()) if isinstance(value, dict) else value)


def _inside(address: str, selection: set[str]) -> bool:
    return not selection or any(address == s or address.startswith(s + ".") for s in selection)


def _nnsight():
    try:
        from nnsight import NNsight
    except ImportError as e:
        raise ValueError(MISSING) from e
    return NNsight


def _raise(e: Exception, edits: bool = False):
    orig = getattr(e, "original", e)  # NNsightException wraps forward()'s exception
    if isinstance(orig, torch.cuda.OutOfMemoryError) or (edits and isinstance(orig, ValueError)):
        raise orig from None
    first = (str(orig).strip().splitlines() or [""])[0][:300]
    raise TraceError(
        f"NNsight could not trace forward(): {type(orig).__name__}: {first}. Check that model(x) runs on "
        "this input outside Neurotome."
    ) from e


def _trace(model: torch.nn.Module, args: tuple, keep: set[str]) -> _Recorder:
    NNsight = _nnsight()
    try:
        with _Recorder(model, keep=keep) as rec, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with NNsight(model).trace(*args):
                pass
    except Exception as e:  # noqa: BLE001 - anything NNsight or forward() raises
        _raise(e)
    return rec


def _structure(model: torch.nn.Module, args: tuple, rec: _Recorder) -> list[dict[str, Any]]:
    """IR ops without values: input ops, leaf calls and the output op, with edges by tensor identity."""
    mods = dict(model.named_modules())
    producer: dict[int, str] = {}
    ops: list[dict[str, Any]] = []

    def add(oid: str, kind: str, t: torch.Tensor | None, parents: list[str], call: _Call | None = None):
        ops.append({
            "id": oid, "type": kind, "func": None,
            "shape": list(t.shape) if t is not None else [],
            "dtype": str(t.dtype) if t is not None else "",
            "device": str(t.device) if t is not None else None,
            "parents": parents, "children": [],
            "modules": call.modules if call else [],
            "module": call.address if call else None,
            "num_params": sum(p.numel() for p in mods[call.address].parameters(recurse=False)) if call else 0,
            "flops": 0, "is_input": kind == "input", "is_output": kind == "output",
        })  # fmt: skip
        if t is not None:
            producer[t.data_ptr()] = oid

    for i, t in enumerate((a for a in args if isinstance(a, torch.Tensor)), start=1):
        add(f"input_{i}", "input", t, [])
    for c in rec.calls:
        prev = [ops[-1]["id"]] if ops else []
        parents = list(dict.fromkeys(producer[p] for p in c.inputs if p in producer)) or prev
        add(c.id, type(mods[c.address]).__name__.lower(), c.out, parents, c)
    out = _output_tensor(rec.output)
    last = [ops[-1]["id"]] if ops else []
    add(
        "output_1",
        "output",
        out,
        [producer[out.data_ptr()]] if out is not None and out.data_ptr() in producer else last,
    )
    by_id = {o["id"]: o for o in ops}
    for o in ops:
        for p in o["parents"]:
            by_id[p]["children"].append(o["id"])
    return ops


def capture(
    model: torch.nn.Module,
    inputs: Any,
    store: RunStore,
    run_id: str,
    max_bytes: int | None = None,
    modules: list[str] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """``capture.capture`` at module granularity: same arguments, budget and selection rules, same IR."""
    model.eval()
    named = dict(model.named_modules())
    selection = set(modules or ())
    if modules:
        _check_modules(modules, named)
    keep = {a for a, m in named.items() if next(m.children(), None) is None and _inside(a, selection)}
    if progress:
        progress(0.12, "Trace")
    args = _args(inputs)
    rec = _trace(model, args, keep)
    ops = _structure(model, args, rec)
    calls = {c.id: c for c in rec.calls}
    ends = [t for t in args if isinstance(t, torch.Tensor)] + [_output_tensor(rec.output)]
    io = {o["id"]: t for o, t in zip([o for o in ops if _io(o)], ends, strict=True)}
    total = sum(t.numel() * 4 for t in io.values() if t is not None)
    over, disk_bytes = False, 0
    n, step = len(ops), max(1, len(ops) // 50)
    for i, o in enumerate(ops, start=1):
        c, unsaved = calls.get(o["id"]), None
        if c is None:  # input / output
            t = io[o["id"]]
            arr = None if t is None else t.detach().to("cpu", torch.float32).numpy()
            o["stats"] = None if arr is None else tensor_stats(arr)
        else:
            arr, o["stats"], c.value = c.value, c.stats, None
            if c.out is not None and arr is None:
                unsaved = "modules"
            elif arr is not None and (over or (max_bytes is not None and total + arr.nbytes > max_bytes)):
                over, unsaved = True, "budget"
            elif arr is not None:
                total += arr.nbytes
        if arr is None and unsaved is None:
            unsaved = "no_tensor"
        if unsaved is None and arr is not None:
            disk_bytes += store.save_tensor(run_id, o["id"], arr)
        del arr
        o["saved"], o["unsaved"] = unsaved is None, unsaved
        if progress and (i % step == 0 or i == n):
            progress(0.2 + 0.7 * i / n, f"Save tensors {i}/{n}")

    body = [o for o in ops if not _io(o)]
    return {
        "model": {
            "class_name": type(model).__name__,
            "num_params": int(sum(p.numel() for p in model.parameters())),
            "num_ops": len(body),
            "flops": 0,
            "activation_bytes": disk_bytes,
            "truncated": over,
            "budget_bytes": max_bytes,
            "capture_modules": list(modules) if modules else None,
            "saved_ops": sum(1 for o in body if o["saved"]),
            "backend": "nnsight",
            "granularity": "module",
        },
        "ops": ops,
        "modules": {
            a: {
                "class_name": type(named[a]).__name__,
                "num_params": int(sum(p.numel() for p in named[a].parameters())),
                "depth": a.count(".") + 1 if a else 0,
            }
            for a in rec.counts
        },
    }


def _io(o: dict[str, Any]) -> bool:
    return bool(o["is_input"] or o["is_output"])


def op_gradients(
    model: torch.nn.Module, inputs: Any, target: int | None, to_logits, objective: dict | None = None
) -> dict[str, Any]:
    """``capture.op_gradients`` at module granularity (plain autograd, the same op ids as ``capture``).
    ``merges`` is always empty: graph shares fall back to parent scores (``share_method: parent``)."""
    model.eval()
    args = _args(_grad_leaf(inputs))
    with _Recorder(model, grads=True) as rec:
        output = model(*args)
    out = _output_tensor(output)
    if out is not None and out.requires_grad:
        out.retain_grad()
    acts = {f"input_{i}": t for i, t in enumerate((a for a in args if isinstance(a, torch.Tensor)), start=1)}
    acts |= {c.id: c.value for c in rec.calls} | {"output_1": out}  # leaf values: pre-in-place clones
    logit = None
    if objective is None:
        logits = to_logits(output)
        if target is None:
            target = int(logits[0].detach().argmax())
        root = logits[0, target]
        logit = float(root.detach())
    else:
        op_id, c = objective["op"], int(objective["channel"])
        unit = acts.get(op_id)
        if unit is None:
            raise ValueError(f"Unknown op {op_id}")
        if unit.dim() < 2 or not 0 <= c < unit.shape[1]:
            raise ValueError(f"Channel {c} out of range for {op_id} with shape {list(unit.shape)}")
        target, root = None, unit[0, c].sum()
    if not root.requires_grad:
        raise ValueError(NO_GRAD)
    root.backward()

    scores: dict[str, float] = {}
    channel_scores: dict[str, list[float]] = {}
    grads = {c.id: c.grad for c in rec.calls}
    for oid, act in acts.items():
        grad = grads[oid] if oid in grads else act.grad if act is not None else None
        if act is None or grad is None or act.shape != grad.shape or not act.is_floating_point():
            continue
        gxa = (act.detach() * grad.detach()).to(torch.float32)
        scores[oid] = float(gxa.sum())
        if gxa.dim() >= 3:  # (B, C, ...) -> per channel
            channel_scores[oid] = gxa[0].flatten(1).sum(1).tolist()
    return {
        "target": target,
        "logit": logit,
        "value": float(root.detach()),
        "scores": scores,
        "channel_scores": channel_scores,
        "merges": {},
    }


@torch.enable_grad()  # as at capture: no_grad lets fused fast paths skip submodule calls
def intervene(
    model: torch.nn.Module,
    inputs: Any,
    edits: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    downstream: Collection[str],
    keep: Collection[str] = (),
) -> dict[str, Any]:
    """``capture.intervene`` on leaf-module calls: a clean pass, then one NNsight pass that sets
    ``env.output = fn(env.output)`` at each edited call, in execution order. The recorder's hooks see an
    edited call before NNsight's swap, so its after value is ``fn(recorded)``; its consumers see the edit."""
    NNsight = _nnsight()
    model.eval()
    args = _args(inputs)
    leaves = {a for a, m in model.named_modules() if next(m.children(), None) is None}
    fns: dict[str, list] = {}
    for op, fn in edits:
        if op.rpartition("@")[0] not in leaves:
            raise ValueError(f"Unknown op {op}")
        fns.setdefault(op, []).append(fn)
    down = set(downstream)
    addresses = {op.rpartition("@")[0] for op in down}
    clean = _trace(model, args, addresses)
    order = [c.id for c in clean.calls]
    for op in fns:
        if op not in order:
            raise ValueError(f"Unknown op {op}")
    before = _values(clean, down)

    def composed(op: str):
        def fn(t):
            if not torch.is_tensor(t):
                raise ValueError(f"{op} returns several tensors; edit one of its consumers")
            new = t
            for f in fns[op]:
                new = f(new)
            if tuple(new.shape) != tuple(t.shape):
                raise ValueError(f"Edit of {op} changed its shape {list(t.shape)} -> {list(new.shape)}")
            return new

        return fn

    todo = [(op, composed(op)) for op in sorted(fns, key=order.index)]
    nn_model: Any = NNsight(model)
    try:
        with _Recorder(model, keep=addresses) as rec, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with nn_model.trace(*args) as tr:
                for op, fn in todo:
                    address, _, call = op.rpartition("@")
                    env = reduce(
                        getattr, address.split("."), nn_model
                    )  # ModuleList children are "0", "1", ...
                    # every edit inside tr.iter: the per-module call tracker only counts while an iter is open
                    for _ in tr.iter[int(call) - 1]:
                        env.output = fn(env.output)
                out = nn_model.output.save()
    except Exception as e:  # noqa: BLE001 - anything NNsight, forward() or an edit raises
        _raise(e, edits=True)
    after = _values(rec, down)
    for op, fn in todo:
        if op in after:
            after[op] = fn(torch.from_numpy(after[op])).numpy()
    nodes, stats, kept = {}, {}, {}
    for op, b in before.items():
        a = after.get(op)
        if a is None or a.shape != b.shape:
            continue
        nodes[op], stats[op] = diff_row(b, a), tensor_stats(a)
        if op in keep:
            kept[op] = a
    return {"nodes": nodes, "stats": stats, "after": kept, "logits_after": out}


def _values(rec: _Recorder, ops: set[str]) -> dict[str, Any]:
    """Float32 arrays of the recorded float calls in ``ops`` (and the output op)."""
    vals = {c.id: c.value for c in rec.calls if c.id in ops and c.value is not None and c.out is not None
            and c.out.is_floating_point()}  # fmt: skip
    out = _output_tensor(rec.output)
    if "output_1" in ops and out is not None and out.is_floating_point():
        vals["output_1"] = out.detach().to("cpu", torch.float32).numpy().copy()
    return vals
