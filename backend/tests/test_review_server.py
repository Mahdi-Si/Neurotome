"""0.3.0 review: the DNS-rebinding guard, NEUROTOME_* parsing and the budget of a run's interventions."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient
from test_pipeline import Tiny

from neurotome import cli
from neurotome.api import create_app, local_host
from neurotome.config import get_settings
from neurotome.runs import RunManager
from neurotome.store import RunStore


def test_local_host():
    for h in ("localhost:7070", "127.0.0.1:7070", "[::1]:7070", "192.168.1.4", "app.localhost:5173", None):
        assert local_host(h), h
    for h in ("evil.example.com", "evil.example.com:7070", "127.0.0.1.nip.io"):
        assert not local_host(h), h
    assert local_host("mybox:7070", ("mybox",))


def test_rebound_host_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROTOME_ALLOWED_HOSTS", "mybox")
    client = TestClient(create_app(RunManager(RunStore(tmp_path / "runs"))))
    assert client.get("/api/runs").status_code == 200  # TestClient's own host
    r = client.get("/api/runs", headers={"host": "attacker.example:7070"})
    assert r.status_code == 403 and "NEUROTOME_ALLOWED_HOSTS" in r.json()["detail"]
    assert client.get("/api/runs", headers={"host": "127.0.0.1:7070"}).status_code == 200
    assert client.get("/api/runs", headers={"host": "mybox:7070"}).status_code == 200


@pytest.mark.parametrize(("mb", "want"), [("512", 512 * 2**20), ("0", None), ("-1", None), ("inf", None)])
def test_capture_budget_env(monkeypatch, mb, want):
    monkeypatch.setenv("NEUROTOME_MAX_CAPTURE_MB", mb)
    assert get_settings().max_capture_bytes == want


def test_bad_env_value_is_a_plain_message(monkeypatch, capsys):
    monkeypatch.setenv("NEUROTOME_PORT", "seventy")
    with pytest.raises(ValueError, match="NEUROTOME_PORT='seventy' is not a number"):
        get_settings()
    assert cli.main(["serve"]) == 1
    assert "is not a number" in capsys.readouterr().err


def test_intervention_budget_is_the_captures_own(tmp_path):
    """A run captured with a tiny budget keeps it for interventions (0 bytes read as the server default)."""
    mgr = RunManager(RunStore(tmp_path / "runs"), max_capture_bytes=None)
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16), max_capture_mb=1e-9)["id"]
    ir = mgr.store.ir(run)
    assert ir["model"]["budget_bytes"] == 0
    op = next(o["id"] for o in ir["ops"] if o["type"] == "conv2d")
    res = mgr.intervene(run, {"target": None, "edits": [{"op": op, "kind": "zero", "channels": None,
                                                         "positions": None}]})
    assert res["saved"] == [] and res["truncated"] and res["budget_bytes"] == 0


def test_reversed_global_range_is_the_same_range():
    import numpy as np

    from neurotome.tensors import tensor_view

    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    fwd, rev = (tensor_view(a, norm="global", lo=x, hi=y) for x, y in ((0.0, 31.0), (31.0, 0.0)))
    assert rev["scale"] == [0.0, 31.0] and rev["maps"] == fwd["maps"] and rev["dots"] == fwd["dots"]


def test_negative_steer_label_uses_the_minus_sign():
    from neurotome.interventions import label

    e = {"op": "x", "kind": "steer", "alpha": -2.0, "direction": {"kind": "channel", "channel": 3},
         "channels": None, "positions": None}
    assert label(e, "block").startswith("Steer −2 × e_3")
