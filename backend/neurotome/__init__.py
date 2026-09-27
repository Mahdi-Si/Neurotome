"""Neurotome: open a PyTorch model and see every signal.

Quick use from a script or notebook::

    import neurotome
    neurotome.open(model, x)          # capture, then serve the UI at http://127.0.0.1:7070
    neurotome.open(lm, ids, tokenizer=tok)   # token labels and, for a causal LM, the logit lens
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"
__all__ = ["open", "__version__"]
_servers: dict[int, tuple[Any, Any]] = {}  # port -> (RunManager, server thread) of open(block=False) calls


def open(
    model: Any,
    inputs: Any,
    name: str | None = None,
    categories: list[str] | None = None,
    port: int | None = None,
    block: bool = True,
    tokenizer: Any = None,
) -> str:
    """Capture ``model(inputs)`` as a new run and serve the UI.

    ``block=False`` starts the server in a background thread (useful in notebooks)
    and returns the run URL immediately; calling ``open`` again on that port adds the run to the running
    server. ``tokenizer`` (``encode`` / ``decode``, e.g. a Hugging Face
    tokenizer) labels token ids and, for a causal LM, enables the logit lens.
    """
    import threading

    import uvicorn

    from .api import create_app
    from .config import get_settings
    from .runs import RunManager

    st = get_settings()
    port = port or st.port
    # a second call reuses the server this process already runs on the port: a new server could not bind
    # it, and the running one would not hold the new run's model
    running = _servers.get(port)
    running = running if running and running[1].is_alive() else None
    mgr = running[0] if running else RunManager.from_settings(st)
    meta = mgr.create_from_objects(model, inputs, name=name, categories=categories, tokenizer=tokenizer)
    url = f"http://{st.host}:{port}/#/runs/{meta['id']}/graph"
    print(f"Neurotome -> {url}")
    if running:
        return url
    server = uvicorn.Server(uvicorn.Config(create_app(mgr), host=st.host, port=port, log_level="warning"))
    if block:
        server.run()
    else:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _servers[port] = (mgr, thread)
    return url
