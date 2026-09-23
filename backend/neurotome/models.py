"""Model loading.

A model is named by a *spec* string so a run can be reloaded after a restart:

* ``torchvision:resnet18``            any name accepted by ``torchvision.models.get_model``
* ``py:examples.custom_model:build``  import ``build`` from a module on ``sys.path`` and call it

Models built in-process (``neurotome.open(model, x)``) have no spec and live only
in memory for the server's lifetime; their captured data persists on disk.

Loading ``py:`` specs imports and runs arbitrary code. The server binds to
localhost by default for this reason — do not expose it on a network.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

SUGGESTED_MODELS = [
    "torchvision:resnet18",
    "torchvision:resnet50",
    "torchvision:mobilenet_v3_small",
    "torchvision:efficientnet_b0",
    "torchvision:vgg16",
    "torchvision:convnext_tiny",
    "torchvision:vit_b_16",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LoadedModel:
    model: nn.Module
    spec: str | None
    weights: str | None
    categories: list[str] | None = None
    # Turns a PIL image into a batched input tensor; None when the model does not take images.
    preprocess: Callable[[Any], torch.Tensor] | None = None
    default_input_shape: list[int] = field(default_factory=lambda: [1, 3, 224, 224])


def _default_image_preprocess(size: int = 224) -> Callable[[Any], torch.Tensor]:
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return lambda img: tf(img.convert("RGB")).unsqueeze(0)


def load_model(spec: str, weights: str | None = None, device: str = "cpu") -> LoadedModel:
    kind, _, rest = spec.partition(":")
    if kind == "torchvision":
        return _load_torchvision(rest, weights, device)
    if kind == "py":
        return _load_python(rest, device)
    raise ValueError(f"Unknown model spec {spec!r}. Use 'torchvision:<name>' or 'py:<module>:<factory>'.")


def _load_torchvision(name: str, weights: str | None, device: str) -> LoadedModel:
    import torchvision

    w = None
    categories = None
    preprocess = _default_image_preprocess()
    if weights:
        enum = torchvision.models.get_model_weights(name)
        w = enum.DEFAULT if weights.upper() == "DEFAULT" else enum[weights]
        categories = list(w.meta.get("categories") or []) or None
        tf = w.transforms()
        preprocess = lambda img: tf(img.convert("RGB")).unsqueeze(0)  # noqa: E731
    model = torchvision.models.get_model(name, weights=w).eval().to(device)
    return LoadedModel(
        model=model,
        spec=f"torchvision:{name}",
        weights=weights,
        categories=categories,
        preprocess=preprocess,
    )


def _load_python(target: str, device: str) -> LoadedModel:
    module_name, _, attr = target.rpartition(":")
    if not module_name:
        raise ValueError("py specs look like 'py:package.module:factory'")
    # console scripts don't put the working directory on sys.path; `python -m` does
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    mod = importlib.import_module(module_name)
    factory = getattr(mod, attr)
    model = factory().eval().to(device)
    shape = list(getattr(mod, "INPUT_SHAPE", [1, 3, 224, 224]))
    categories = getattr(mod, "CATEGORIES", None)
    preprocess = getattr(mod, "preprocess", None)
    if preprocess is None and len(shape) == 4 and shape[1] in (1, 3):
        preprocess = _default_image_preprocess(shape[-1])
    return LoadedModel(
        model=model,
        spec=f"py:{target}",
        weights=None,
        categories=list(categories) if categories else None,
        preprocess=preprocess,
        default_input_shape=shape,
    )


def to_logits(output: Any) -> torch.Tensor:
    """Reduce a model's output to a (batch, classes) tensor that attribution can target."""
    if isinstance(output, torch.Tensor):
        out = output
    elif hasattr(output, "logits"):
        out = output.logits
    elif isinstance(output, dict):
        out = next(v for v in output.values() if isinstance(v, torch.Tensor))
    elif isinstance(output, (tuple, list)):
        out = next(v for v in output if isinstance(v, torch.Tensor))
    else:
        raise TypeError(f"Cannot find a tensor in model output of type {type(output).__name__}")
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.dim() == 3:  # (batch, tokens, vocab): explain the last position
        out = out[:, -1, :]
    if out.dim() > 2:
        out = out.flatten(1)
    return out
