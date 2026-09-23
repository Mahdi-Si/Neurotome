"""Attribution: which inputs and which internal units drove one objective.

The objective is a logit (``target``) or an internal unit: Σ over positions of one channel of a
module's output.

1. **Input attribution** (Captum): a signed map over the input for the chosen method.
2. **Internal attribution**: gradient x activation for *every* op in the graph from one backward pass
   (via TorchLens), plus each parent's grad x input through every merge (edge shares at residual adds, cats).
3. **Layer methods** (Captum layer IG / conductance, per channel) and a channel's neuron conductance.
4. **Agreement** between methods: top-k Jaccard / overlap and Spearman rank correlation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .capture import _first_tensor, op_gradients
from .models import to_logits
from .tensors import pool2d

METHODS = {
    "gxa": "Gradient x input",
    "saliency": "Saliency |grad|",
    "ig": "Integrated gradients",
}
LAYER_METHODS = {"layer_ig": "Layer integrated gradients", "layer_cond": "Layer conductance"}
NEURON_LABEL = "Neuron conductance (Σ over positions)"
MERGE_METHOD = "Gradient x input at merge inputs"

Progress = Callable[[float, str], None]


def logit_label(target: int, categories: list[str] | None) -> str:
    name = categories[target] if categories and target < len(categories) else None
    return f"logit[{target}]" + (f" · {name}" if name else "")


def unit_label(path: str, channel: int) -> str:
    return f"{path} · ch {channel} · Σ spatial"


def objective_forward(
    model, site: tuple[str, int] | None, channel: int | None
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int | None]:
    """(fwd, target) for Captum. site None -> (logits, None: the caller's target).
    Otherwise a forward hook on the module keeps its ``call``-th output of each forward pass and
    fwd(x) returns out[:, channel] summed over positions as (B, 1); target 0.
    The hook is registered inside fwd and removed in a finally block (never left on the model)."""
    if site is None:
        return (lambda x: to_logits(model(x))), None
    address, call = site
    module = model.get_submodule(address)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        seen: list[Any] = []
        handle = module.register_forward_hook(lambda _m, _i, out: seen.append(_clone(out)))
        try:
            model(x)
        finally:
            handle.remove()
        out = _first_tensor(seen[call - 1])
        return out[:, channel].reshape(out.shape[0], -1).sum(1, keepdim=True)

    return fwd, 0


def _clone(out: Any) -> Any:
    """A copy of a module output, so later in-place ops (ReLU(inplace), ``+=``) cannot change it."""
    if torch.is_tensor(out):
        return out.clone()
    return type(out)(map(_clone, out)) if type(out) in (tuple, list) else out


def _shielded(fwd, layer: torch.nn.Module):
    """fwd with a hook, registered last on ``layer`` at call time, that hands the rest of the model a copy of
    its output: Captum's hook output (a leaf for layer IG) then never meets the in-place ops after it."""

    def wrapped(x):
        handle = layer.register_forward_hook(lambda _m, _i, out: _clone(out))
        try:
            return fwd(x)
        finally:
            handle.remove()

    return wrapped


def _batches(steps: int) -> tuple[int, int]:
    """(internal batch size, number of batches) for Riemann / IG steps."""
    size = max(1, min(steps, 16))
    return size, math.ceil(steps / size)


def _reporting(fwd, n: int, label: str, progress: Progress | None):
    """Wrap fwd to report "<label> i/n" after each of its first n calls (Captum's delta pass is silent)."""
    if progress is None:
        return fwd
    calls = [0]

    def wrapped(x):
        out = fwd(x)
        calls[0] += 1
        if calls[0] <= n:
            progress(calls[0] / n, f"{label} {calls[0]}/{n}")
        return out

    return wrapped


def input_map(attr: torch.Tensor, delta: float | None) -> dict[str, Any]:
    """A raw input attribution (same shape as x) as a signed map pooled to <= 24x24 and scaled to [-1, 1]."""
    a = attr.detach()[0].to("cpu", torch.float32).numpy()
    if a.ndim == 3:  # (C, H, W) -> sum over channels
        a = a.sum(axis=0)
    if a.ndim == 1:
        a = a[None, :]
    a = a.reshape(a.shape[0], -1) if a.ndim > 2 else a
    grid = pool2d(a, 24, 24)
    m = float(np.abs(grid).max()) or 1.0
    total = float(attr.detach().sum())
    return {"map": np.round(grid / m, 3).tolist(), "total": total, "convergence_delta": delta}


def input_attribution(
    model, x: torch.Tensor, target: int, method: str, steps: int, fwd=None, progress: Progress | None = None
) -> dict[str, Any]:
    from captum.attr import InputXGradient, IntegratedGradients, Saliency

    fwd = fwd or (lambda x: to_logits(model(x)))
    x = x.detach().clone().requires_grad_(True)
    delta = None
    if method == "ig":
        size, n = _batches(steps)
        attr, d = IntegratedGradients(_reporting(fwd, n, "IG batch", progress)).attribute(
            x,
            baselines=torch.zeros_like(x),
            target=target,
            n_steps=steps,
            internal_batch_size=size,
            return_convergence_delta=True,
        )
        delta = float(d.abs().max())
    elif method == "saliency":
        attr = Saliency(fwd).attribute(x, target=target, abs=True)
    elif method == "gxa":
        attr = InputXGradient(fwd).attribute(x, target=target)
    else:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")
    return input_map(attr, delta)


def attribute(
    model,
    inputs: Any,
    method: str = "ig",
    target: int | None = None,
    steps: int = 32,
    categories: list[str] | None = None,
    unit_ops: set[str] | None = None,
    objective: dict[str, Any] | None = None,
    site: tuple[str, int] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """``unit_ops``: op ids eligible for the top-units list (default: every op with a channel axis).
    The run manager passes ops that own parameters, so BatchNorm/ReLU copies of the same channel
    do not crowd out distinct units.
    ``objective`` ``{"op", "channel"}`` explains that unit instead of a logit (``target`` is ignored);
    ``site`` is its module call (``graph.module_site``), which the input methods hook.
    The result also carries ``channel_scores`` for the run manager, which keeps them server-side."""
    model.eval()
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    report = progress or (lambda _f, _m: None)

    probs = None
    if objective is None:
        with torch.no_grad():
            logits = to_logits(model(x))
            probs = torch.softmax(logits[0].float(), dim=0)
        if target is None:
            target = int(logits[0].argmax())
    elif site is None:
        raise ValueError("A unit objective needs its module site")
    else:
        target = None

    report(0.02, "Node gradients")
    internal = op_gradients(model, inputs, target, to_logits, objective)

    input_attr = None
    if torch.is_tensor(x) and x.is_floating_point():
        fwd, t = objective_forward(model, site if objective else None, objective and objective["channel"])
        sub = (lambda f, m: report(0.1 + 0.88 * f, m)) if progress else None
        input_attr = input_attribution(model, x, target if t is None else t, method, steps, fwd, sub)

    top_units = []
    for op_id, chans in internal["channel_scores"].items():
        if unit_ops is not None and op_id not in unit_ops:
            continue
        for c, v in enumerate(chans):
            top_units.append((abs(v), op_id, c, v))
    top_units.sort(reverse=True)
    if objective is None:
        info = {"kind": "logit", "index": target}
        label = logit_label(target, categories)
    else:
        info = {"kind": "unit", "op": objective["op"], "channel": int(objective["channel"])}
        label = unit_label(objective["op"], info["channel"])
    return {
        "method": method,
        "method_label": METHODS.get(method, method),
        "node_method": "Gradient x activation",
        "merge_method": MERGE_METHOD,
        "objective": info,
        "objective_label": label,
        "target": target,
        "target_label": (categories[target] if target is not None and categories and target < len(categories)
                         else None),
        "prob": float(probs[target]) if probs is not None else None,
        "logit": internal["logit"],
        "value": internal["value"],
        "steps": steps if method == "ig" else None,
        "input": input_attr,
        "scores": internal["scores"],
        "merges": internal["merges"],
        "channel_scores": internal["channel_scores"],
        "top_units": [{"op": o, "channel": c, "value": v} for _, o, c, v in top_units[:12]],
    }  # fmt: skip


def layer_attribution(
    model, x: torch.Tensor, layer: torch.nn.Module, method: str, fwd, target: int | None, steps: int,
    progress: Progress | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Captum layer IG / conductance on ``layer``'s output, summed per channel (axis 1)."""
    from captum.attr import LayerConductance, LayerIntegratedGradients

    if method not in LAYER_METHODS:
        raise ValueError(f"Unknown layer method {method!r}; choose one of {sorted(LAYER_METHODS)}")
    cls, label = (LayerIntegratedGradients, "Layer IG batch") if method == "layer_ig" else (
        LayerConductance, "Conductance batch")  # fmt: skip
    size, n = _batches(steps)
    x = x.detach().clone().requires_grad_(True)
    attr, delta = cls(_shielded(_reporting(fwd, n, label, progress), layer), layer).attribute(
        x,
        baselines=torch.zeros_like(x),
        target=target,
        n_steps=steps,
        internal_batch_size=size,
        return_convergence_delta=True,
    )
    a = _first_tensor(attr).detach()[0].float()
    chans = (a.flatten(1).sum(1) if a.dim() >= 2 else a).cpu()
    order = chans.abs().argsort(descending=True, stable=True)[:12]
    return {
        "channels": chans.tolist(),
        "top": [{"channel": int(c), "value": float(chans[c])} for c in order],
        "total": float(chans.sum()),
        "convergence_delta": float(delta.abs().max()),
    }


def channel_conductance(
    model, x: torch.Tensor, layer: torch.nn.Module, channel: int, fwd, target: int, steps: int,
    progress: Progress | None = None,
) -> torch.Tensor:  # fmt: skip
    """Raw (same shape as x) Σ over positions j of channel ``channel`` of neuron conductance, baseline 0,
    Riemann midpoints α_k = (k + ½) / steps. Captum's NeuronConductance refuses aggregated neurons."""
    size, n = _batches(steps)
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps
    x = x.detach()
    acc = torch.zeros_like(x[0])
    for i in range(n):
        a = alphas[i * size : (i + 1) * size].reshape(-1, *([1] * (x.dim() - 1)))
        kept: list[Any] = []

        def keep(_m, _i, out, kept=kept):  # grad w.r.t. the output as computed; the model runs on a copy
            kept.append(out)
            return _clone(out)

        handle = layer.register_forward_hook(keep)
        try:
            with torch.enable_grad():
                xs = (a * x).requires_grad_(True)
                f = fwd(xs)[:, target]
                y = _first_tensor(kept[0])
                g = torch.autograd.grad(f.sum(), y, retain_graph=True)[0].detach()
                s = (g[:, channel] * y[:, channel]).sum()
                acc += torch.autograd.grad(s, xs)[0].sum(0)
        finally:
            handle.remove()
        if progress:
            progress((i + 1) / n, f"Conductance batch {i + 1}/{n}")
    return x * acc / steps


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    r = np.empty(a.size)
    r[np.argsort(a, kind="stable")] = np.arange(a.size)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / counts)[inv]


def agreement(
    vectors: dict[str, np.ndarray], labels: dict[str, str], basis: str, k_frac: float = 0.1
) -> dict[str, Any]:
    """Pairwise agreement of attribution vectors, ranked by |value|: Jaccard and overlap of the top-k sets,
    Spearman rank correlation (average ranks for ties; None when either vector is constant)."""
    methods = list(vectors)
    n = min(np.asarray(v).size for v in vectors.values())
    mags = [np.abs(np.asarray(vectors[m], dtype=np.float64).ravel()[:n]) for m in methods]
    k = max(1, math.ceil(k_frac * n))
    tops = [set(np.argsort(-a, kind="stable")[:k].tolist()) for a in mags]
    ranks = [_avg_ranks(a) for a in mags]
    size = len(methods)
    jac = [[1.0] * size for _ in range(size)]
    ovl = [[1.0] * size for _ in range(size)]
    rho: list[list[float | None]] = [[1.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            inter = len(tops[i] & tops[j])
            jac[i][j] = jac[j][i] = inter / len(tops[i] | tops[j])
            ovl[i][j] = ovl[j][i] = inter / k
            ri, rj = ranks[i] - ranks[i].mean(), ranks[j] - ranks[j].mean()
            den = math.sqrt(float((ri * ri).sum() * (rj * rj).sum()))
            rho[i][j] = rho[j][i] = float((ri * rj).sum() / den) if den > 0 else None
    return {
        "basis": basis,
        "methods": methods,
        "labels": [labels.get(m, m) for m in methods],
        "n": int(n),
        "k": k,
        "jaccard": jac,
        "overlap": ovl,
        "spearman": rho,
    }
