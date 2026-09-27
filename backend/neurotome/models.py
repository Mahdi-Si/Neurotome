"""Model loading.

A model is named by a *spec* string so a run can be reloaded after a restart:

* ``torchvision:resnet18``            any name accepted by ``torchvision.models.get_model``
* ``hf:gpt2``, ``hf:<local dir>``     a Hugging Face model (``hf`` extra) with its tokenizer / image processor
* ``py:examples.custom_model:build``  import ``build`` from a module on ``sys.path`` and call it; the
  module may also define ``INPUT_SHAPE``, ``CATEGORIES``, ``TOKENIZER``, ``preprocess(img)``, ``OUTPUT_KIND``
  and a dataset (``list_samples()`` / ``load_sample(ref)`` / ``target_windows(index)``, see
  ``examples/signal_model.py``)

Models built in-process (``neurotome.open(model, x)``) have no spec and live only
in memory for the server's lifetime; their captured data persists on disk.

Loading ``py:`` specs imports and runs arbitrary code. The server binds to
localhost by default for this reason — do not expose it on a network.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
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
    "hf:gpt2",
    "hf:Qwen/Qwen2.5-0.5B",
    "hf:google/vit-base-patch16-224",
]
TEXT_MODELS = ["hf:gpt2", "hf:Qwen/Qwen2.5-0.5B"]
HF_MISSING = "Hugging Face models need the hf extra: pip install 'neurotome[hf]'"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Vocab(Sequence[str]):
    """Token labels decoded on demand, so a 150k-token vocabulary is never materialised."""

    def __init__(self, tokenizer: Any, size: int):
        self.tokenizer, self.size = tokenizer, size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> str:  # type: ignore[override]  # ints only, no slices
        if not isinstance(i, int) or not 0 <= i < self.size:
            raise IndexError(i)
        return self.tokenizer.decode([i])


@dataclass
class LoadedModel:
    model: nn.Module
    spec: str | None
    weights: str | None
    categories: Sequence[str] | None = None  # a list, or a Vocab for models with a tokenizer
    # Turns a PIL image into a batched input tensor; None when the model does not take images.
    preprocess: Callable[[Any], torch.Tensor] | None = None
    default_input_shape: list[int] = field(default_factory=lambda: [1, 3, 224, 224])
    # (mean, std) per channel that preprocess normalised with, so the input preview can undo it
    norm: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    # encode(text) -> ids, decode(ids) -> str, optional pad_token_id / eos_token_id (HF tokenizers qualify)
    tokenizer: Any | None = None
    lm: bool = False  # a causal LM with a tokenizer: the logit lens applies
    # what preprocess does to an image, in words (shown next to the raw and the model input); None: unknown
    steps: list[str] | None = None
    # "logits": the output summary is a softmax top-5; "values": top-5 by raw value (scores, regression)
    output_kind: str = "logits"
    # a py: module's dataset hook: ref -> {input, raw, blocks, time, steps, info} (examples/signal_model.py)
    load_sample: Callable[[str], dict[str, Any]] | None = None


def _default_transform(size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _default_image_preprocess(size: int = 224) -> Callable[[Any], torch.Tensor]:
    tf = _default_transform(size)
    return lambda img: tf(img.convert("RGB")).unsqueeze(0)


def _vec(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(x):.3g}" for x in v) + ")"


def _normalize_step(mean: Sequence[float], std: Sequence[float]) -> str:
    return f"Normalize (x − mean) / std · mean {_vec(mean)} · std {_vec(std)}"


def transform_steps(tf: Any) -> list[str]:
    """A torchvision image transform in words: Compose lists, the weights' ImageClassification preset, or its
    repr for anything else. Every image path converts to RGB first and adds the batch axis last."""
    from torchvision import transforms as T

    how = lambda m: getattr(m, "value", str(m))  # noqa: E731 - InterpolationMode.BILINEAR -> "bilinear"
    if isinstance(tf, T.Compose):
        return [s for t in tf.transforms for s in transform_steps(t)]
    if hasattr(tf, "crop_size") and hasattr(tf, "resize_size"):  # torchvision weights: ImageClassification
        c = tf.crop_size[0]
        resize = f"Resize shorter side to {tf.resize_size[0]} px ({how(tf.interpolation)})"
        return [resize, f"Center crop {c}×{c}", "Scale to [0, 1] (÷ 255)", _normalize_step(tf.mean, tf.std)]
    if isinstance(tf, T.Resize):
        s = tf.size
        if isinstance(s, int) or len(s) == 1:
            short = s if isinstance(s, int) else s[0]
            return [f"Resize shorter side to {short} px ({how(tf.interpolation)})"]
        return [f"Resize to {s[0]}×{s[1]} ({how(tf.interpolation)})"]
    if isinstance(tf, T.CenterCrop):
        return [f"Center crop {tf.size[0]}×{tf.size[1]}"]
    if isinstance(tf, T.ToTensor):
        return ["Scale to [0, 1] (÷ 255)"]
    if isinstance(tf, T.Normalize):
        return [_normalize_step(tf.mean, tf.std)]
    return [" ".join(repr(tf).split())]


def processor_steps(p: Any) -> list[str]:
    """A Hugging Face image processor in words (the attributes its __call__ applies, in that order)."""
    get = lambda d, k: d.get(k) if isinstance(d, dict) else getattr(d, k, None)  # noqa: E731 - dict or SizeDict
    steps = []
    if getattr(p, "do_resize", False) and (size := getattr(p, "size", None)) is not None:
        how = getattr(getattr(p, "resample", None), "name", "").lower()
        h, w, short = get(size, "height"), get(size, "width"), get(size, "shortest_edge")
        what = f"to {h}×{w}" if h and w else f"shorter side to {short} px" if short else str(size)
        steps.append(f"Resize {what}" + (f" ({how})" if how else ""))
    if getattr(p, "do_center_crop", False) and (crop := getattr(p, "crop_size", None)) is not None:
        steps.append(f"Center crop {get(crop, 'height')}×{get(crop, 'width')}")
    if getattr(p, "do_rescale", False):
        f = float(getattr(p, "rescale_factor", 1 / 255))
        steps.append("Scale to [0, 1] (÷ 255)" if abs(f - 1 / 255) < 1e-9 else f"Rescale × {f:.6g}")
    if getattr(p, "do_normalize", False):
        steps.append(_normalize_step(p.image_mean, p.image_std))
    return steps


def load_model(spec: str, weights: str | None = None, device: str = "cpu") -> LoadedModel:
    """Random-init weights are seeded, so a spec reloads to the same model it was captured with
    (attribution after a restart, patching from another run of the same spec)."""
    kind, _, rest = check_spec(spec).partition(":")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if kind == "torchvision":
            return _load_torchvision(rest, weights, device)
        if kind == "hf":
            return _load_hf(rest, weights, device)
        return _load_python(rest, device)


def check_spec(spec: str) -> str:
    """``spec`` when its prefix is known, else ValueError."""
    if spec.partition(":")[0] not in ("torchvision", "hf", "py"):
        raise ValueError(f"Unknown model spec {spec!r}. Use 'torchvision:<name>', 'hf:<repo_id>' or "
                         "'py:<module>:<factory>'.")  # fmt: skip
    return spec


def _load_torchvision(name: str, weights: str | None, device: str) -> LoadedModel:
    import torchvision

    w = None
    categories = None
    preprocess = _default_image_preprocess()
    norm = (IMAGENET_MEAN, IMAGENET_STD)
    steps = transform_steps(_default_transform())
    if weights:
        enum = torchvision.models.get_model_weights(name)
        w = enum["DEFAULT" if weights.upper() == "DEFAULT" else weights]
        categories = list(w.meta.get("categories") or []) or None
        tf = w.transforms()
        preprocess = lambda img: tf(img.convert("RGB")).unsqueeze(0)  # noqa: E731
        steps = transform_steps(tf)
        if hasattr(tf, "mean") and hasattr(tf, "std"):
            norm = (tuple(tf.mean), tuple(tf.std))
    model = torchvision.models.get_model(name, weights=w).eval().to(device)
    return LoadedModel(
        model=model,
        spec=f"torchvision:{name}",
        weights=weights,
        categories=categories,
        preprocess=preprocess,
        norm=norm,
        steps=["Convert to RGB", *steps],
    )


def is_lm(model: nn.Module) -> bool:
    return type(model).__name__.endswith(("ForCausalLM", "LMHeadModel"))


def is_hf(model: nn.Module) -> bool:
    """A transformers ``PreTrainedModel``, checked by class name so transformers is never imported."""
    return any(c.__name__ == "PreTrainedModel" for c in type(model).__mro__)


def _prepare_hf(model: nn.Module) -> nn.Module:
    """Eager attention (so softmax ops exist) and no KV cache (else the trace has extra output ops)."""
    with suppress(Exception):
        model.set_attn_implementation("eager")  # type: ignore[operator]
    model.config.use_cache = False  # type: ignore[union-attr]
    return model.eval()


def vocab_size(model: nn.Module) -> int | None:
    """Rows of the unembedding (padded vocabularies included); None without an LM head (a classifier)."""
    head = None
    with suppress(Exception):
        head = model.get_output_embeddings()  # type: ignore[operator]
    if head is None:
        return None
    n = getattr(head, "out_features", None) or getattr(getattr(model, "config", None), "vocab_size", None)
    return int(n) if n else None


def _load_hf(repo: str, weights: str | None, device: str) -> LoadedModel:
    try:
        import transformers
    except ImportError as e:
        raise ValueError(HF_MISSING) from e
    config = transformers.AutoConfig.from_pretrained(repo)
    arch = (getattr(config, "architectures", None) or [""])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:  # no (known) architecture: a causal LM by its model type, else the base model
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as CAUSAL

        cls = transformers.AutoModelForCausalLM if config.model_type in CAUSAL else transformers.AutoModel
    # ponytail: float32 only; bf16 / device_map for >16 GB models belongs to v0.4 (Scale)
    kw: dict[str, Any] = {"attn_implementation": "eager", "dtype": torch.float32}
    if weights:
        model = cls.from_pretrained(repo, **kw)
    else:
        make = getattr(cls, "from_config", None) or cls._from_config  # pyright: ignore[reportAttributeAccessIssue]  # Auto class, else a model class
        model = make(config, **kw)
    loaded = LoadedModel(_prepare_hf(model).to(device), f"hf:{repo}", weights)
    tok = proc = None
    try:
        tok = transformers.AutoTokenizer.from_pretrained(repo)
    except Exception:  # noqa: BLE001 - no tokenizer files: an image (or tensor-only) model
        with suppress(Exception):
            proc = transformers.AutoImageProcessor.from_pretrained(repo)
    id2label = getattr(config, "id2label", None) or {}
    labels = [str(id2label[i]) for i in range(len(id2label))]
    labels = None if all(n == f"LABEL_{i}" for i, n in enumerate(labels)) else labels
    if tok is not None:
        v = vocab_size(model)
        loaded.tokenizer, loaded.lm, loaded.default_input_shape = tok, is_lm(model), [1, 16]
        loaded.categories = Vocab(tok, v) if v else labels
    elif proc is not None:
        size = getattr(config, "image_size", None) or 224
        loaded.preprocess = lambda img: proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
        loaded.norm = (tuple(proc.image_mean), tuple(proc.image_std))
        loaded.steps = ["Convert to RGB", *processor_steps(proc)]
        loaded.default_input_shape = [1, 3, size, size]
        loaded.categories = labels
    return loaded


def _py_module(target: str) -> tuple[Any, str, str]:
    """(module, module name, factory name) of a ``py:`` spec's target, imported, factory not called."""
    module_name, _, attr = target.rpartition(":")
    if not module_name:
        raise ValueError("py specs look like 'py:package.module:factory'")
    # console scripts don't put the working directory on sys.path; `python -m` does
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return importlib.import_module(module_name), module_name, attr


def _sample_row(s: Any) -> dict[str, Any]:
    if not isinstance(s, dict) or not isinstance(s.get("ref"), str):
        raise ValueError(f"list_samples() rows must be dicts with a str 'ref', got {s!r:.80}")
    g = s.get("group")
    return {"ref": s["ref"], "label": str(s.get("label") or s["ref"]), "group": None if g is None else str(g)}


# ponytail: cached for the server's life per spec; a dataset that changes on disk needs a restart
@lru_cache(maxsize=16)
def list_samples(spec: str) -> list[dict[str, Any]]:
    """The samples a ``py:`` module's ``list_samples()`` offers (import only: the model is not built); [] for
    other specs and modules without the hook. Import or hook errors raise (the API reports them)."""
    kind, _, rest = check_spec(spec).partition(":")
    if kind != "py":
        return []
    fn: Callable[[], list[Any]] | None = getattr(_py_module(rest)[0], "list_samples", None)
    return [_sample_row(s) for s in fn()] if callable(fn) else []


def target_windows(spec: str, index: int) -> Any:
    """What a ``py:`` module's ``target_windows(index)`` returns (import only: the model is not built),
    unchecked (``runs.check_windows`` checks it against the run's input); [] for other specs and modules
    without the hook."""
    kind, _, rest = check_spec(spec).partition(":")
    fn = getattr(_py_module(rest)[0], "target_windows", None) if kind == "py" else None
    return fn(index) if callable(fn) else []


def _load_python(target: str, device: str) -> LoadedModel:
    mod, module_name, attr = _py_module(target)
    factory = getattr(mod, attr)
    torch.manual_seed(0)  # again: the first import may have drawn random numbers, a cached module does not
    model = factory()
    model = (_prepare_hf(model) if is_hf(model) else model.eval()).to(device)
    tok = getattr(mod, "TOKENIZER", None)
    shape = list(getattr(mod, "INPUT_SHAPE", [1, 16] if tok else [1, 3, 224, 224]))
    categories = getattr(mod, "CATEGORIES", None)
    v = vocab_size(model) if tok else None
    preprocess = getattr(mod, "preprocess", None)
    norm = None
    steps = [f"Custom: {module_name}.preprocess"] if preprocess is not None else None
    if preprocess is None and len(shape) == 4 and shape[1] in (1, 3):
        preprocess = _default_image_preprocess(shape[-1])
        norm = (IMAGENET_MEAN, IMAGENET_STD)
        steps = ["Convert to RGB", *transform_steps(_default_transform(shape[-1]))]
    return LoadedModel(
        model=model,
        spec=f"py:{target}",
        weights=None,
        categories=list(categories) if categories else Vocab(tok, v) if v else None,
        preprocess=preprocess,
        default_input_shape=shape,
        norm=norm,
        tokenizer=tok,
        lm=tok is not None and is_lm(model),
        steps=steps,
        output_kind=getattr(mod, "OUTPUT_KIND", "logits"),
        load_sample=getattr(mod, "load_sample", None),
    )


def to_logits(output: Any) -> torch.Tensor:
    """Reduce a model's output to a (batch, classes) tensor that attribution can target."""
    out: Any = None
    if isinstance(output, torch.Tensor):
        out = output
    elif getattr(output, "logits", None) is not None:
        out = output.logits
    if not isinstance(out, torch.Tensor):
        seq = list(output.values()) if isinstance(output, dict) else output
        seq = seq if isinstance(seq, (tuple, list)) else []
        out = next((v for v in seq if isinstance(v, torch.Tensor)), None)
    if out is None:
        raise ValueError(
            f"The model returned {type(output).__name__}, which holds no tensor. Neurotome explains a "
            "tensor output: return logits, a dict/tuple/list that holds a tensor (the first one is used), or "
            "an object with .logits (Hugging Face ModelOutput)."
        )
    if not out.is_floating_point():
        raise ValueError(f"The model output is {out.dtype}; attribution needs float logits.")
    if out.dim() == 0:
        out = out.reshape(1, 1)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.dim() == 3:  # (batch, tokens, vocab): explain the last position; a masked LM too
        out = out[:, -1, :]
    if out.dim() > 2:
        out = out.flatten(1)
    return out
