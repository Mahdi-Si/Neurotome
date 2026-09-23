"""Attribution: which inputs and which internal units drove one output.

Two layers of answer, computed together:

1. **Input attribution** (Captum): a signed map over the input for the chosen method.
2. **Internal attribution**: gradient x activation for *every* op in the graph from one
   backward pass (via TorchLens), which gives node scores, edge shares at merges
   (residual vs identity) and the top channels. This is the cheap, first-order
   estimate; more faithful per-layer methods (layer IG, conductance, activation
   patching) are on the roadmap and slot in behind the same result schema.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .capture import op_gradients
from .models import to_logits
from .tensors import pool2d

METHODS = {
    "gxa": "Gradient x input",
    "saliency": "Saliency |grad|",
    "ig": "Integrated gradients",
}


def _forward_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: to_logits(model(x))


def input_attribution(model, x: torch.Tensor, target: int, method: str, steps: int) -> dict[str, Any]:
    from captum.attr import InputXGradient, IntegratedGradients, Saliency

    fwd = _forward_fn(model)
    x = x.detach().clone().requires_grad_(True)
    delta = None
    if method == "ig":
        attr, d = IntegratedGradients(fwd).attribute(
            x,
            baselines=torch.zeros_like(x),
            target=target,
            n_steps=steps,
            internal_batch_size=max(1, min(steps, 16)),
            return_convergence_delta=True,
        )
        delta = float(d.abs().max())
    elif method == "saliency":
        attr = Saliency(fwd).attribute(x, target=target, abs=True)
    elif method == "gxa":
        attr = InputXGradient(fwd).attribute(x, target=target)
    else:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(METHODS)}")

    a = attr.detach()[0].to("cpu", torch.float32).numpy()
    if a.ndim == 3:  # (C, H, W) -> sum over channels
        a = a.sum(axis=0)
    if a.ndim == 1:
        a = a[None, :]
    a = a.reshape(a.shape[0], -1) if a.ndim > 2 else a
    grid = pool2d(a, 24, 24)
    m = float(np.abs(grid).max()) or 1.0
    return {
        "map": np.round(grid / m, 3).tolist(),
        "total": float(attr.detach().sum()),
        "convergence_delta": delta,
    }


def attribute(
    model,
    inputs: Any,
    method: str = "ig",
    target: int | None = None,
    steps: int = 32,
    categories: list[str] | None = None,
    unit_ops: set[str] | None = None,
) -> dict[str, Any]:
    """``unit_ops``: op ids eligible for the top-units list (default: every op with a channel axis).
    The run manager passes ops that own parameters, so BatchNorm/ReLU copies of the same channel
    do not crowd out distinct units."""
    model.eval()
    x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs

    with torch.no_grad():
        logits = to_logits(model(x))
        probs = torch.softmax(logits[0].float(), dim=0)
    if target is None:
        target = int(logits[0].argmax())

    internal = op_gradients(model, inputs, target, to_logits)

    input_attr = None
    if torch.is_tensor(x) and x.is_floating_point():
        input_attr = input_attribution(model, x, target, method, steps)

    top_units = []
    for op_id, chans in internal["channel_scores"].items():
        if unit_ops is not None and op_id not in unit_ops:
            continue
        for c, v in enumerate(chans):
            top_units.append((abs(v), op_id, c, v))
    top_units.sort(reverse=True)
    return {
        "method": method,
        "method_label": METHODS.get(method, method),
        "node_method": "Gradient x activation",
        "target": target,
        "target_label": (categories[target] if categories and target < len(categories) else None),
        "prob": float(probs[target]),
        "logit": internal["logit"],
        "steps": steps if method == "ig" else None,
        "input": input_attr,
        "scores": internal["scores"],
        "top_units": [{"op": o, "channel": c, "value": v} for _, o, c, v in top_units[:12]],
    }
