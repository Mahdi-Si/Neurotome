"""The Input view: the raw sample kept with the run, the preprocessing in words, the model input's ranges."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from neurotome.api import create_app
from neurotome.runs import RunManager
from neurotome.store import RunStore

REPO = Path(__file__).resolve().parents[2]


def _image(fmt: str, size=(300, 200)) -> bytes:
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def app(tmp_path):
    mgr = RunManager(RunStore(tmp_path / "runs"))
    return mgr, TestClient(create_app(mgr))


def test_uploaded_image_keeps_its_raw_copy_and_steps(app):
    mgr, client = app
    data = _image("JPEG")
    uid = mgr.save_upload("cat.jpg", data)
    sample = {"kind": "upload", "upload_id": uid, "name": "cat.jpg"}
    meta = mgr.create_from_spec("torchvision:resnet18", None, sample)
    run = meta["id"]
    raw = meta["input"]["raw"]
    assert raw == {"file": "input_raw.jpg", "width": 300, "height": 200, "format": "JPEG", "bytes": len(data)}
    assert (mgr.store.run_dir(run) / "input_raw.jpg").read_bytes() == data
    assert meta["input"]["steps"][:3] == ["Convert to RGB", "Resize shorter side to 256 px (bilinear)",
                                          "Center crop 224×224"]
    info = client.get(f"/api/runs/{run}/input").json()
    assert info["shape"] == [1, 3, 224, 224] and info["picture"] and info["normalised"]
    assert len(info["channels"]) == 3 and all(c["min"] <= c["mean"] <= c["max"] for c in info["channels"])
    x = np.asarray(mgr.store.load_tensor(run, info["op"]))
    assert info["channels"][1]["max"] == pytest.approx(float(x[0, 1].max()), rel=1e-6)
    r = client.get(f"/api/runs/{run}/input/raw")
    assert r.status_code == 200 and r.content == data and r.headers["content-type"] == "image/jpeg"
    pic = client.get(f"/api/runs/{run}/input/model.png")
    assert pic.status_code == 200 and Image.open(io.BytesIO(pic.content)).size == (224, 224)


def test_a_format_browsers_cannot_show_is_kept_as_png(app):
    mgr, _ = app
    uid = mgr.save_upload("scan.tif", _image("TIFF", (64, 48)))
    meta = mgr.create_from_spec("torchvision:resnet18", None, {"kind": "upload", "upload_id": uid})
    raw = meta["input"]["raw"]
    assert raw["file"] == "input_raw.png" and raw["format"] == "TIFF"
    assert (raw["width"], raw["height"]) == (64, 48)
    assert Image.open(mgr.store.run_dir(meta["id"]) / "input_raw.png").size == (64, 48)


def test_random_and_text_samples(app, monkeypatch):
    mgr, client = app
    monkeypatch.syspath_prepend(str(REPO))
    run = mgr.create_from_spec("torchvision:resnet18", None, {"kind": "random"})["id"]
    info = client.get(f"/api/runs/{run}/input").json()
    assert info["raw"] is None and info["steps"][0].startswith("None: a random input")
    assert not info["normalised"] and client.get(f"/api/runs/{run}/input/raw").status_code == 404
    assert client.get(f"/api/runs/{run}/input/model.png").status_code == 200  # the noise, stretched
    pytest.importorskip("transformers")
    text = mgr.create_from_spec("py:examples.tiny_lm:build", None, {"kind": "text", "text": "the cat"})["id"]
    info = client.get(f"/api/runs/{text}/input").json()
    assert info["steps"] == ["Tokenize with CharTokenizer → token ids"] and not info["picture"]
    assert info["sample"]["tokens"] == list("the cat") and len(info["channels"]) == 1
    assert client.get(f"/api/runs/{text}/input/model.png").status_code == 404


def test_old_runs_open(app):
    mgr, client = app
    run = mgr.create_from_spec("torchvision:resnet18", None, {"kind": "random"})["id"]
    meta = mgr.store.meta(run)
    del meta["input"]
    mgr.store.write_json(run, "meta.json", meta)
    info = client.get(f"/api/runs/{run}/input").json()
    assert info["raw"] is None and info["steps"] is None and info["picture"]
