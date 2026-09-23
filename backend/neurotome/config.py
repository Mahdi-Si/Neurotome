"""Runtime configuration, read from environment variables with sensible local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    host: str
    port: int
    device: str
    # Refuse to store more than this many bytes of activations for a single run.
    max_capture_bytes: int
    # Models (+ inputs) kept loaded for attribution; least recently used is evicted beyond this.
    max_live_models: int


def get_settings() -> Settings:
    return Settings(
        runs_dir=Path(os.environ.get("NEUROTOME_RUNS_DIR", "./runs")).resolve(),
        host=os.environ.get("NEUROTOME_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEUROTOME_PORT", "7070")),
        device=os.environ.get("NEUROTOME_DEVICE", "cpu"),
        max_capture_bytes=int(float(os.environ.get("NEUROTOME_MAX_CAPTURE_MB", "2048")) * 1024 * 1024),
        max_live_models=int(os.environ.get("NEUROTOME_MAX_LIVE_MODELS", "2")),
    )
