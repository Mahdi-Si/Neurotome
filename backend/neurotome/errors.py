"""One place that turns exceptions into user-facing messages (HTTP status + plain text, never a traceback)."""

from __future__ import annotations

import torch


class TraceError(ValueError):
    """TorchLens could not trace forward(). HTTP 422."""


class ModelUnavailable(RuntimeError):
    """A run's model is not loaded and cannot be reloaded. HTTP 409."""


NO_GRAD = ("The objective has no gradient: forward() leaves autograd between the input and the output "
           "(for example .numpy(), .item(), .detach() or torch.no_grad()), or the model has no trainable "
           "parameters and an integer input, so gradient methods cannot run on this model. The graph and "
           "tensors are still valid.")  # fmt: skip


_NO_GRAD_TEXTS = ("does not require grad", "appears to not have been used in the graph")


def oom_message(e: BaseException) -> str:
    first = (str(e).strip().splitlines() or [type(e).__name__])[0]
    return (f"CUDA ran out of memory ({first}). Capture on the CPU instead (neurotome serve --device cpu "
            "or NEUROTOME_DEVICE=cpu), lower the capture budget, or capture fewer modules.")  # fmt: skip


def describe(e: BaseException) -> tuple[int, str]:
    """(HTTP status, message) for any exception; the message never contains a traceback."""
    msg = str(e) or type(e).__name__
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 413, oom_message(e)
    if isinstance(e, TraceError):
        return 422, msg
    if isinstance(e, ModelUnavailable):
        return 409, msg
    if isinstance(e, ValueError):
        return 400, msg
    # ponytail: message sniffing; torch raises a plain RuntimeError when autograd is cut in forward()
    if isinstance(e, RuntimeError) and any(s in str(e) for s in _NO_GRAD_TEXTS):
        return 400, NO_GRAD
    return 500, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
