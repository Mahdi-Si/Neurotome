"""v0.3 steering and greedy generation on a tiny GPT-2 (offline: ``py:examples.tiny_lm:build``)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from test_attr_m3 import finish, hooked
from test_jobs_m3 import poll
from test_pipeline import Tiny

from neurotome.api import create_app
from neurotome.graph import module_site
from neurotome.interventions import _apply, edit_fn, normalise
from neurotome.models import to_logits
from neurotome.runs import NOT_LM_GEN, RunManager
from neurotome.store import RunStore

pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[2]
TINY_LM = "py:examples.tiny_lm:build"


@pytest.fixture()
def lm(tmp_path, monkeypatch):
    """Three text runs of the tiny LM (same spec and weights) and block 0's output op."""
    monkeypatch.syspath_prepend(str(REPO))
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    mgr = RunManager(RunStore(tmp_path / "runs"), max_live_models=4)
    texts = ("the cat sat on the mat", "a dog ran far", "hello world")
    ids = [mgr.create_from_spec(TINY_LM, None, {"kind": "text", "text": t})["id"] for t in texts]
    ir = mgr.store.ir(ids[0])
    block0 = next(o["id"] for o in ir["ops"] if module_site(ir, o["id"]) == ("transformer.h.0", 1))
    return mgr, ids, ir, block0


def steer(op, alpha, direction, **scope):
    return {"edits": [{"op": op, "kind": "steer", "alpha": alpha, "direction": direction, **scope}]}


def test_contrast_direction(lm):
    mgr, (a, b, c), ir, op = lm
    A, B = (np.asarray(mgr.store.load_tensor(r, op), np.float64) for r in (b, c))
    for pool, want in (("mean", B.mean((0, 1)) - A.mean((0, 1))), ("last", B[0, -1] - A[0, -1])):
        norm = normalise(steer(op, 1.0, {"kind": "contrast", "run_a": b, "run_b": c, "pool": pool}), ir)
        v = mgr.resolve(a, norm)[0][0]
        assert v.shape == (32,) and np.allclose(v, want, atol=1e-5)
    res = mgr.intervene(a, norm)
    assert res["labels"] == [f"Steer +1 × ({c} − {b}, last) · transformer.h.0 · all"]
    live = mgr.live(a)
    ref = hooked(
        live.loaded.model,
        live.loaded.model.transformer.h[0],
        live.inputs,
        lambda o: _apply(o, lambda t: t + torch.from_numpy(v)),
    )
    assert res["after"]["logit"] == pytest.approx(float(to_logits(ref)[0, res["target"]]), abs=1e-4)
    assert res["nodes"][op]["l2"] > 0 and res["target_label"] == mgr.store.meta(a)["output"][0]["label"]


def test_alpha_zero_and_clean_greedy(lm):
    mgr, (a, *_), ir, op = lm
    norm = normalise(steer(op, 0.0, {"kind": "channel", "channel": 3}), ir)
    assert all(n["l2"] == 0 for n in mgr.intervene(a, norm)["nodes"].values())
    progress = []
    res = mgr.generate(a, norm, 8, progress=lambda f, m: progress.append(m))
    assert res["clean"] == res["steered"] and res["diverge"] is None and res["n"] == 8
    assert res["method_label"] == (
        "Greedy decoding · 8 new tokens · edits as forward hooks on module outputs · no KV cache"
    )
    meta = mgr.store.meta(a)
    assert res["prompt"] == meta["sample"]["tokens"] and res["labels"] == [
        "Steer +0 × e_3 · transformer.h.0 · all"
    ]
    assert progress[0] == "Clean 1/8" and progress[-1] == "Intervened 8/8"
    live = mgr.live(a)
    x, tok, ids = live.inputs, live.loaded.tokenizer, []
    for _ in range(8):
        with torch.no_grad():
            nxt = int(to_logits(live.loaded.model(x))[0].argmax())
        ids.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    clean = res["clean"]
    assert (
        clean["ids"] == ids
        and clean["text"] == tok.decode(ids)
        and clean["tokens"] == [tok.decode([i]) for i in ids]
    )
    assert len(clean["top"]) == 8 and all(len(s) == 5 for s in clean["top"])
    assert all(
        s[0]["id"] == i and s[0]["prob"] >= s[1]["prob"] for s, i in zip(clean["top"], ids, strict=True)
    )
    assert (mgr.store.iv_dir(a, res["hash"]) / "generate_8.json").is_file()


def test_large_steer_changes_generation(lm):
    mgr, (a, *_), ir, op = lm
    norm = normalise(steer(op, 50.0, {"kind": "channel", "channel": 3}, positions="last"), ir)
    res = mgr.generate(a, norm, 4)
    clean, steered = res["clean"]["top"][0], res["steered"]["top"][0]
    assert [t["id"] for t in clean] != [t["id"] for t in steered]
    live = mgr.live(a)
    fn = edit_fn(norm["edits"][0], None)
    ref = hooked(live.loaded.model, live.loaded.model.transformer.h[0], live.inputs, lambda o: _apply(o, fn))
    assert steered[0]["id"] == int(to_logits(ref)[0].argmax())
    assert res["labels"] == ["Steer +50 × e_3 · transformer.h.0 · last token"]


def test_generate_jobs_and_errors(lm):
    mgr, (a, *_), ir, op = lm
    c = TestClient(create_app(mgr))
    jobs = f"/api/runs/{a}/jobs"
    body = {
        "kind": "generate",
        "interventions": steer(op, 2.0, {"kind": "channel", "channel": 1}),
        "max_new_tokens": 3,
    }
    job = poll(c, c.post(jobs, json=body).json())
    assert job["status"] == "done", job["error"]
    assert job["result"]["n"] == 3 and len(job["result"]["steered"]["ids"]) == 3
    again = c.post(jobs, json=body).json()
    assert again["status"] == "done" and again["result"] == job["result"]
    assert c.post(jobs, json={**body, "max_new_tokens": 65}).status_code == 422
    assert c.post(jobs, json={**body, "max_new_tokens": 0}).status_code == 422
    r = c.post(
        jobs,
        json={
            "kind": "generate",
            "interventions": {"edits": [{"op": op, "kind": "resample", "source_run": a}]},
        },
    )
    assert r.status_code == 400 and r.json()["detail"] == (
        "Generation supports zero, mean over positions, constant, scale and steer edits "
        "(resample values have the "
        "prompt's length)"
    )
    r = c.post(
        jobs,
        json={"kind": "generate", "interventions": {"edits": [{"op": op, "kind": "mean", "over": "runs"}]}},
    )
    assert r.status_code == 400 and "(mean over runs values" in r.json()["detail"]
    add = next(
        o["id"]
        for o in ir["ops"]
        if o["type"] == "add"
        and o["modules"][-1:] == ["transformer.h.0.attn:1"]
        and module_site(ir, o["id"]) is None
    )
    r = c.post(jobs, json={"kind": "generate", "interventions": {"edits": [{"op": add, "kind": "zero"}]}})
    assert r.status_code == 400 and r.json()["detail"] == (
        f"Generation edits {add} through a forward hook on its module; {add} is not the output of a module "
        "(pick its module node)"
    )
    assert c.post(jobs, json={"kind": "generate"}).json()["detail"] == "generate jobs need interventions"
    with pytest.raises(
        ValueError, match=re.escape("The prompt (22 tokens) plus 64 new tokens exceeds the model's")
    ):
        cfg = mgr.live(a).loaded.model.config
        cfg.n_positions = 80  # the tiny LM has 128 positions
        try:
            mgr.generate(a, normalise(body["interventions"], ir), 64)
        finally:
            cfg.n_positions = 128

    other = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))["id"]
    r = c.post(
        f"/api/runs/{other}/jobs",
        json={"kind": "generate", "interventions": {"edits": [{"op": "conv2d_3_5", "kind": "zero"}]}},
    )
    assert r.status_code == 400 and r.json()["detail"] == NOT_LM_GEN
    done = finish(
        mgr,
        mgr.job(
            other, {"kind": "intervene", "interventions": {"edits": [{"op": "conv2d_3_5", "kind": "zero"}]}}
        ),
    )
    assert done["status"] == "done"


def test_apply_output_kinds():
    from collections import namedtuple

    from transformers.modeling_outputs import BaseModelOutput

    t, double = torch.ones(2), lambda x: x * 2
    mo = _apply(BaseModelOutput(last_hidden_state=t), double)
    assert isinstance(mo, BaseModelOutput) and mo.last_hidden_state.tolist() == [2, 2]
    assert mo[0] is mo.last_hidden_state
    assert _apply({"n": 1, "x": t}, double)["x"].tolist() == [2, 2]
    Pair = namedtuple("Pair", "a b")
    out = _apply(Pair(None, t), double)
    assert isinstance(out, Pair) and out.b.tolist() == [2, 2]
    out = _apply((1, t, t), double)
    assert type(out) is tuple and out[1].tolist() == [2, 2] and out[2] is t
    with pytest.raises(ValueError, match="Cannot edit a module output of type dict: it holds no tensor"):
        _apply({"n": 1}, double)


def test_generate_through_model_output(lm):
    """``transformer`` returns a ModelOutput: its hook edits last_hidden_state (never a silent no-op)."""
    mgr, (a, *_), ir, _op = lm
    op = next(o["id"] for o in ir["ops"] if module_site(ir, o["id"]) == ("transformer", 1))
    norm = normalise({"edits": [{"op": op, "kind": "zero"}]}, ir)
    res = mgr.generate(a, norm, 3)
    live = mgr.live(a)
    want = to_logits(hooked(live.loaded.model, live.loaded.model.transformer, live.inputs,
                            lambda o: _apply(o, torch.zeros_like)))[0]  # fmt: skip
    assert res["steered"]["ids"][0] == int(want.argmax()) and res["diverge"] == 0


def test_generate_job_nnsight(tmp_path, monkeypatch):
    pytest.importorskip("nnsight")
    monkeypatch.syspath_prepend(str(REPO))
    mgr = RunManager(RunStore(tmp_path / "runs"))
    a = mgr.create_from_spec(TINY_LM, None, {"kind": "text", "text": "the cat sat"}, backend="nnsight")["id"]
    site = "transformer.h.0.mlp.c_proj@1"
    edits = steer(site, 50.0, {"kind": "channel", "channel": 3}, positions="last")
    body = {"kind": "generate", "max_new_tokens": 3, "interventions": edits}
    job = finish(mgr, mgr.job(a, body))
    assert job["status"] == "done", job["error"]
    live = mgr.live(a)
    fn = edit_fn(normalise(body["interventions"], mgr.store.ir(a))["edits"][0], None)
    ref = hooked(live.loaded.model, live.loaded.model.transformer.h[0].mlp.c_proj, live.inputs, fn)
    assert job["result"]["steered"]["ids"][0] == int(to_logits(ref)[0].argmax())
    again = mgr.job(a, body)
    assert again.status == "done" and again.result == job["result"]
    done = finish(mgr, mgr.job(a, {"kind": "intervene", "interventions": body["interventions"]}))
    assert done["status"] == "done" and done["result"]["method_label"].startswith("Intervention · NNsight")
