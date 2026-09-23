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


def get_settings() -> Settings:
    return Settings(
        runs_dir=Path(os.environ.get("DISSECT_RUNS_DIR", "./runs")).resolve(),
        host=os.environ.get("DISSECT_HOST", "127.0.0.1"),
        port=int(os.environ.get("DISSECT_PORT", "7070")),
        device=os.environ.get("DISSECT_DEVICE", "cpu"),
        max_capture_bytes=int(float(os.environ.get("DISSECT_MAX_CAPTURE_MB", "2048")) * 1024 * 1024),
    )
