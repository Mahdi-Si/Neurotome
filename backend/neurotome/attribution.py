"""Attribution: which inputs and which internal units drove one objective.

The objective is a logit (``target``) or an internal unit: Σ over positions of one channel of a
module's output.

1. **Input attribution** (Captum): a signed map over a float input for the chosen method, or, for token ids,
   one score per token from the same method applied to the embedding layer's output.
2. **Internal attribution**: gradient x activation for *every* op in the graph from one backward pass
   (via TorchLens), plus each parent's grad x input through every merge (edge shares at residual adds, cats).
3. **Layer methods** (Captum layer IG / conductance, per channel) and a channel's neuron conductance.
4. **Agreement** between methods: top-k Jaccard / overlap and Spearman rank correlation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .capture import _out, op_gradients
from .interventions import channel_axis, per_channel
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


def logit_label(target: int, categories: Sequence[str] | None) -> str:
    name = categories[target] if categories and target < len(categories) else None
    return f"logit[{target}]" + (f" · {name}" if name else "")


def unit_label(path: str, channel: int) -> str:
    return f"{path} · ch {channel} · Σ positions"


def objective_forward(
    model, site: tuple[str, int] | None, channel: int | None
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int | None]:
    """(fwd, target) for Captum. site None -> (logits, None: the caller's target).
    Otherwise a forward hook on the module keeps its ``call``-th output of each forward pass and
    fwd(x) returns channel ``channel`` (``channel_axis``) summed over positions as (B, 1); target 0.
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
        out = _out(seen[call - 1])
        unit = out.select(channel_axis(out.dim()), cast(int, channel))
        return unit.reshape(out.shape[0], -1).sum(1, keepdim=True)

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


def input_map(attr: torch.Tensor, delta: float | None, layout: dict[str, Any] | None = None
              ) -> dict[str, Any]:  # fmt: skip
    """A raw input attribution (same shape as x) as a signed map scaled to [-1, 1]: a sequence ((1, T, C) or
    (1, C, T)) as a channel x time map (``series_map``; ``layout``: the run's meta.input: blocks, time, raw),
    anything else pooled to <= 24x24 (images summed over their channels)."""
    a = attr.detach()[0].to("cpu", torch.float32).numpy()
    total = float(attr.detach()[0].sum())  # sample 0, like the map
    if a.ndim == 2:
        return series_map(a, time_axis(attr.shape, layout), layout or {}, delta, total)
    if a.ndim == 3:  # (C, H, W) -> sum over channels
        a = a.sum(axis=0)
    if a.ndim == 1:
        a = a[None, :]
    a = a.reshape(a.shape[0], -1) if a.ndim > 2 else a
    grid = pool2d(a, 24, 24)
    m = float(np.abs(grid).max()) or 1.0
    return {"map": np.round(grid / m, 3).tolist(), "total": total, "convergence_delta": delta}


SERIES_COLS, SERIES_ROWS = 512, 160  # time columns, channel rows: each axis is pooled only above this


def time_axis(shape: Sequence[int], layout: dict[str, Any] | None) -> int:
    """The time axis (1 or 2) of a (1, A, B) input: meta.input.time's, else 1 when blocks group the last axis
    (load_sample's (1, T, C)), else the longer axis."""
    info = layout or {}
    if (t := info.get("time")) and t.get("axis") in (1, 2):
        return int(t["axis"])
    if info.get("blocks"):
        return 1
    return 1 if shape[1] >= shape[2] else 2  # ponytail: a guess (time is usually longer); `time` settles it


def _block(name: str, start: int, stop: int, unit: str | None = None) -> dict[str, Any]:
    return {"name": name, "start": start, "stop": stop, "labels": None, "unit": unit}


def series_blocks(info: dict[str, Any], axis: int, chans: int) -> list[dict[str, Any]]:
    """The channel groups of a sequence input with time on ``axis``: meta.input.blocks (time on axis 1), else
    one block per raw signal channel when there are ``chans`` of them (named like the channel), else
    ``input``."""
    if axis == 1 and (blocks := info.get("blocks")):
        return blocks
    raw = info.get("raw") or {}
    chs = raw.get("channels", []) if raw.get("kind") == "signal" else []
    names = [(c["name"], c.get("unit")) for c in chs]
    return [_block(n, i, i + 1, u) for i, (n, u) in enumerate(names)] if len(names) == chans else [
        _block("input", 0, chans)]  # fmt: skip


def series_map(a: np.ndarray, axis: int, info: dict[str, Any], delta: float | None, total: float
               ) -> dict[str, Any]:  # fmt: skip
    """Sample 0 of a sequence attribution as rows = channels, columns = time: sums over ``stride`` steps
    (only above 512) and ``row_stride`` channels (only above 160), scaled to [-1, 1] (``scale``: the divisor).
    ``blocks``: ``series_blocks`` with their Σ a, Σ|a| and
    share of Σ|a|; ``time_totals``: the signed Σ of each column; ``time``: step 0 and the step length in
    seconds on the run's clock (meta.input.time, or the raw rate when the input is the raw signal)."""
    m = (a.T if axis == 1 else a).astype(np.float64)  # (C, T)
    chans, steps = m.shape
    ts, rs = math.ceil(steps / SERIES_COLS), math.ceil(chans / SERIES_ROWS)
    grid = np.add.reduceat(np.add.reduceat(m, np.arange(0, chans, rs), 0), np.arange(0, steps, ts), 1)
    top = float(np.abs(grid).max()) or 1.0
    per = np.abs(m).sum(1)
    whole = float(per.sum()) or 1.0
    raw = info.get("raw") or {}
    raw = raw if raw.get("kind") == "signal" else {}
    blocks = series_blocks(info, axis, chans)
    t, label = info.get("time"), raw.get("time_label") or "s"
    if t and t.get("axis") == axis:
        time = {"t0": float(t["t0"]), "step_s": float(t["step_s"]), "label": label}
    elif raw and all(c.get("length") == steps for c in raw["channels"]):  # the input is the raw signal
        time = {"t0": float(raw["t0"]), "step_s": 1 / float(raw["fs"]), "label": label}
    else:
        time = None
    short = lambda v: float(f"{v:.6g}")  # noqa: E731
    out = []
    for b in sorted(blocks, key=lambda b: b["start"]):
        part = slice(b["start"], b["stop"])
        out.append({**_block(b["name"], b["start"], b["stop"], b.get("unit")), "labels": b.get("labels"),
                    "sum": short(m[part].sum()), "abs": short(per[part].sum()),
                    "share": float(per[part].sum()) / whole})  # fmt: skip
    return {
        "kind": "series",
        # 3 significant digits, not decimals: an unpooled map has many small cells whose sum still counts
        "map": [[float(f"{v:.3g}") for v in row] for row in (grid / top).tolist()],
        "scale": top,  # map * scale = Σ a of each cell
        "total": total,
        "convergence_delta": delta,
        "channels": chans,
        "row_stride": rs,
        "steps": steps,
        "stride": ts,
        "time": time,
        "blocks": out,
        "time_totals": [short(v) for v in grid.sum(0)],
    }


def input_attribution(
    model, x: torch.Tensor, target: int | None, method: str, steps: int, fwd=None,
    progress: Progress | None = None, layout: dict[str, Any] | None = None,
) -> dict[str, Any]:  # fmt: skip
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
    return input_map(attr, delta, layout)


def input_note(x: Any, method: str) -> str | None:
    """Why input attribution is skipped for ``x``, or None when it runs."""
    if not torch.is_tensor(x):
        return "Input attribution needs a tensor input; node scores still cover every op."
    if x.is_floating_point():
        return None
    return (f"{METHODS.get(method, method)} needs a float input; this input is {x.dtype} and the model has "
            "no embedding layer to attribute instead, so input attribution is skipped. Node scores (gradient "
            "× activation) still cover every op.")  # fmt: skip


def embedding_layer(model) -> tuple[str, nn.Module] | None:
    """(address, module) of the input embeddings: ``get_input_embeddings()``, else the first nn.Embedding."""
    layer = None
    with suppress(Exception):
        layer = model.get_input_embeddings()
    if not isinstance(layer, nn.Module):
        layer = next((m for m in model.modules() if isinstance(m, nn.Embedding)), None)
    return next(((a, m) for a, m in model.named_modules() if m is layer), None) if layer is not None else None


def token_attribution(
    model, x: torch.Tensor, target: int | None, method: str, steps: int, fwd, layer: nn.Module, address: str,
    baseline_id: int = 0, progress: Progress | None = None,
) -> dict[str, Any]:  # fmt: skip
    """One score per token of an integer input: the method applied to the embedding layer's output, summed
    over the hidden dimension (saliency: the L2 norm of the gradient, unsigned)."""
    from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

    delta, signed, base = None, True, None
    if method == "ig":
        size, n = _batches(steps)
        attr, d = LayerIntegratedGradients(_reporting(fwd, n, "IG batch", progress), layer).attribute(
            x,
            baselines=torch.full_like(x, baseline_id),
            target=target,
            n_steps=steps,
            internal_batch_size=size,
            return_convergence_delta=True,
        )
        scores, delta, base = _out(attr)[0].sum(-1), float(cast(torch.Tensor, d).abs().max()), baseline_id
        label = f"Layer IG on {address} · baseline token {baseline_id}"
    elif method == "gxa":
        scores = _out(LayerGradientXActivation(fwd, layer).attribute(x, target=target))[0].sum(-1)
        label = f"Gradient x embedding on {address}"
    elif method == "saliency":
        grad = LayerGradientXActivation(fwd, layer, multiply_by_inputs=False).attribute(x, target=target)
        scores, signed, label = _out(grad)[0].norm(dim=-1), False, f"Saliency ‖grad‖ on {address}"
    else:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")
    vals = [float(f"{v:.6g}") for v in scores.detach().float().reshape(-1).cpu().tolist()]
    return {
        "scores": vals,
        "signed": signed,
        "total": sum(vals),
        "convergence_delta": delta,
        "layer": address,
        "baseline_id": base,
        "method_label": label,
    }


def attribute(
    model,
    inputs: Any,
    method: str = "ig",
    target: int | None = None,
    steps: int = 32,
    categories: Sequence[str] | None = None,
    unit_ops: set[str] | None = None,
    objective: dict[str, Any] | None = None,
    site: tuple[str, int] | None = None,
    progress: Progress | None = None,
    baseline_id: int = 0,
    grads=op_gradients,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``unit_ops``: op ids eligible for the top-units list (default: every op with a channel axis).
    The run manager passes ops that own parameters, so BatchNorm/ReLU copies of the same channel
    do not crowd out distinct units.
    ``objective`` ``{"op", "channel"}`` explains that unit instead of a logit (``target`` is ignored);
    ``site`` is its module call (``graph.module_site``), which the input methods hook.
    Integer inputs (token ids) get ``tokens`` (the method on the embedding layer, IG from ``baseline_id``)
    instead of an input map. ``grads`` is the capture backend's ``op_gradients``; ``layout`` the run's
    meta.input (blocks / time / raw of a sequence input, see ``input_map``).
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
    internal = grads(model, inputs, target, to_logits, objective)

    input_attr, tokens, note = None, None, input_note(x, method)
    emb = embedding_layer(model) if note and torch.is_tensor(x) else None
    if note is None or emb:
        channel = int(objective["channel"]) if objective else None
        fwd, t = objective_forward(model, site if objective else None, channel)
        sub = (lambda f, m: report(0.1 + 0.88 * f, m)) if progress else None
        tgt = target if t is None else t
        if emb:
            note = None
            tokens = token_attribution(model, x, tgt, method, steps, fwd, emb[1], emb[0], baseline_id, sub)
        else:
            input_attr = input_attribution(model, x, tgt, method, steps, fwd, sub, layout)

    top_units = []
    for op_id, chans in internal["channel_scores"].items():
        if unit_ops is not None and op_id not in unit_ops:
            continue
        for c, v in enumerate(chans):
            top_units.append((abs(v), op_id, c, v))
    top_units.sort(reverse=True)
    if objective is None:
        info = {"kind": "logit", "index": target}
        label = logit_label(cast(int, target), categories)  # set above when objective is None
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
        "input_note": note,
        "tokens": tokens,
        "scores": internal["scores"],
        "merges": internal["merges"],
        "channel_scores": internal["channel_scores"],
        "top_units": [{"op": o, "channel": c, "value": v} for _, o, c, v in top_units[:12]],
    }  # fmt: skip


def layer_attribution(
    model, x: torch.Tensor, layer: torch.nn.Module, method: str, fwd, target: int | None, steps: int,
    progress: Progress | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Captum layer IG / conductance on ``layer``'s output, summed per channel (``channel_axis``)."""
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
    a = _out(attr).detach()[0].float()
    chans = (per_channel(a) if a.dim() >= 2 else a).cpu()
    order = chans.abs().argsort(descending=True, stable=True)[:12]
    return {
        "channels": chans.tolist(),
        "top": [{"channel": int(c), "value": float(chans[c])} for c in order],
        "total": float(chans.sum()),
        "convergence_delta": float(cast(torch.Tensor, delta).abs().max()),
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
                y = _out(kept[0])
                g = torch.autograd.grad(f.sum(), y, retain_graph=True)[0].detach()
                ax = channel_axis(y.dim())
                s = (g.select(ax, channel) * y.select(ax, channel)).sum()
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
    # a zero is in no top set: ties among zeros would otherwise fill it by index and look like agreement
    tops = [{i for i in np.argsort(-a, kind="stable")[:k].tolist() if a[i] > 0} for a in mags]
    ranks = [_avg_ranks(a) for a in mags]
    size = len(methods)
    jac = [[1.0] * size for _ in range(size)]
    ovl = [[1.0] * size for _ in range(size)]
    rho: list[list[float | None]] = [[1.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            inter = len(tops[i] & tops[j])
            union = len(tops[i] | tops[j])
            jac[i][j] = jac[j][i] = inter / union if union else 1.0  # both all zero
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
