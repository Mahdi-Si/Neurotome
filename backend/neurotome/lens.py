"""Logit lens: the residual stream after every block, read through the model's final norm and unembedding.

Plain forward hooks, one forward pass, no TorchLens: row ``embed`` is block 0's input, row ``<block>`` its
output.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .capture import _out
from .models import to_logits

BLOCKS = ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers", "transformer.layers")
NORMS = ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm", "model.final_layernorm",
         "model.decoder.final_layer_norm", "transformer.norm_f")  # fmt: skip


def unsupported(model: nn.Module) -> str:
    return (f"Logit lens needs a causal LM whose final norm and unembedding it can find (GPT-2, Llama, Qwen, "
            f"Mistral, GPT-NeoX, Phi); {type(model).__name__} is not supported.")  # fmt: skip


def lens_parts(model: nn.Module):
    """(blocks, (norm_path, norm), (head_path, head)); ValueError when any is missing."""
    subs = dict(model.named_modules())
    path = next((p for p in BLOCKS if isinstance(m := subs.get(p), nn.ModuleList) and len(m)), None)
    norm = next((p for p in NORMS if p in subs), None)
    head = None
    get = getattr(model, "get_output_embeddings", None)
    if callable(get):
        out = get()
        head = next((p for p, m in subs.items() if m is out), None) if isinstance(out, nn.Module) else None
    if path is None or norm is None or head is None:
        raise ValueError(unsupported(model))
    blocks = [(f"{path}.{i}", b) for i, b in enumerate(subs[path])]  # type: ignore[arg-type]
    return blocks, (norm, subs[norm]), (head, subs[head])


def logit_lens(
    model: nn.Module, x: torch.Tensor, target: int | None, tokenizer: Any, k: int = 5,
    max_positions: int = 64, progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """``LensResult``: top-k tokens and the target's probability and rank per layer and position, for the
    last ``max_positions`` positions. ``target`` None: the model's own top prediction at the last position."""
    report = progress or (lambda _f, _m: None)
    blocks, (norm_path, norm), (head_path, head) = lens_parts(model)
    resid: list[torch.Tensor] = []

    def pre(_m, args, kwargs):
        resid.append(args[0] if args else kwargs["hidden_states"])

    hooks = [blocks[0][1].register_forward_pre_hook(pre, with_kwargs=True)]
    keep = lambda _m, _i, out: resid.append(_out(out))  # noqa: E731
    hooks += [b.register_forward_hook(keep) for _, b in blocks]
    report(0.1, "Lens forward")
    try:
        with torch.no_grad():
            last = to_logits(model(x))[0]
    finally:
        for h in hooks:
            h.remove()
    if len(resid) != len(blocks) + 1:
        raise ValueError(unsupported(model))
    vocab = last.shape[-1]
    target = int(last.argmax()) if target is None else int(target)
    if not 0 <= target < vocab:
        raise ValueError(f"Target token {target} is outside the vocabulary (0–{vocab - 1})")
    names: dict[int, str] = {}

    def token(i: int) -> str:
        if i not in names:
            names[i] = tokenizer.decode([i]) if tokenizer is not None else str(i)
        return names[i]

    def cells(vals: torch.Tensor, ids: torch.Tensor) -> list[dict[str, Any]]:
        return [{"id": int(j), "token": token(int(j)), "prob": round(float(v), 4)}
                for v, j in zip(vals, ids, strict=True)]  # fmt: skip

    pos0 = max(0, x.shape[-1] - max_positions)
    top, target_prob, target_rank = [], [], []
    for i, h in enumerate(resid):
        report(0.2 + 0.78 * i / len(resid), f"Project layer {i + 1}/{len(resid)}")
        with torch.no_grad():
            p = torch.softmax(head(norm(h))[0, pos0:].float(), -1)  # (positions, vocab)
        vals, ids = p.topk(min(k, vocab), -1)
        top.append([cells(v, j) for v, j in zip(vals, ids, strict=True)])
        pt = p[:, target]
        target_prob.append([round(float(v), 4) for v in pt])
        target_rank.append([int(r) for r in (p > pt[:, None]).sum(-1) + 1])
    return {
        "method_label": f"Logit lens · {norm_path} + {head_path}",
        "layers": ["embed"] + [name for name, _ in blocks],
        "pos0": pos0,
        "tokens": [token(int(i)) for i in x[0, pos0:].tolist()],
        "k": k,
        "top": top,
        "target": {"id": target, "token": token(target)},
        "target_prob": target_prob,
        "target_rank": target_rank,
    }
