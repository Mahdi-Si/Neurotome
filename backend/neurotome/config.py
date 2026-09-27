"""Runtime configuration, read from environment variables with sensible local defaults."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    host: str
    port: int
    device: str
    # Refuse to store more than this many bytes of activations for a single run; None = unlimited.
    max_capture_bytes: int | None
    # Models (+ inputs) kept loaded for attribution; least recently used is evicted beyond this.
    max_live_models: int
    # Auto capture backend: NNsight (module outputs) for models with more parameters than this.
    nnsight_params: int
    # Host names the server answers besides localhost, IP literals and ``host`` (DNS-rebinding guard).
    allowed_hosts: tuple[str, ...] = ()


def _number(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None


def get_settings() -> Settings:
    mb = _number("NEUROTOME_MAX_CAPTURE_MB", "2048")
    return Settings(
        runs_dir=Path(os.environ.get("NEUROTOME_RUNS_DIR", "./runs")).resolve(),
        host=os.environ.get("NEUROTOME_HOST", "127.0.0.1"),
        port=int(_number("NEUROTOME_PORT", "7070")),
        device=os.environ.get("NEUROTOME_DEVICE", "cpu"),
        max_capture_bytes=int(mb * 1024 * 1024) if 0 < mb < math.inf else None,  # <= 0 or inf: unlimited
        max_live_models=max(1, int(_number("NEUROTOME_MAX_LIVE_MODELS", "2"))),
        nnsight_params=int(_number("NEUROTOME_NNSIGHT_PARAMS", "1e9")),
        allowed_hosts=tuple(
            h.strip().lower() for h in os.environ.get("NEUROTOME_ALLOWED_HOSTS", "").split(",") if h.strip()
        ),
    )
