"""Dissect: open a PyTorch model and see every signal.

Quick use from a script or notebook::

    import dissect
    dissect.open(model, x)          # capture, then serve the UI at http://127.0.0.1:7070
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["open", "__version__"]


def open(
    model: Any,
    inputs: Any,
    name: str | None = None,
    categories: list[str] | None = None,
    port: int | None = None,
    block: bool = True,
) -> str:
    """Capture ``model(inputs)`` as a new run and serve the UI.

    ``block=False`` starts the server in a background thread (useful in notebooks)
    and returns the run URL immediately.
    """
    import threading

    import uvicorn

    from .api import create_app
    from .config import get_settings
    from .runs import RunManager
    from .store import RunStore

    st = get_settings()
    mgr = RunManager(RunStore(st.runs_dir), st.device, st.max_capture_bytes)
    meta = mgr.create_from_objects(model, inputs, name=name, categories=categories)
    port = port or st.port
    url = f"http://{st.host}:{port}/#/runs/{meta['id']}/graph"
    print(f"Dissect -> {url}")
    server = uvicorn.Server(uvicorn.Config(create_app(mgr), host=st.host, port=port, log_level="warning"))
    if block:
        server.run()
    else:
        threading.Thread(target=server.run, daemon=True).start()
    return url
