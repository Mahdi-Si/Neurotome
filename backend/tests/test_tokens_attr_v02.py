"""v0.2 token attribution: Layer IG / gxa / saliency on the embedding layer for token-id inputs."""

from __future__ import annotations

import pytest
import torch
from test_hf_v02 import TINY_LM, mgr  # noqa: F401 - mgr is a fixture
from test_jobs_m3 import follow

from neurotome.attribution import embedding_layer
from neurotome.runs import _baseline_id

pytest.importorskip("transformers")
TEXT = {"kind": "text", "text": "the cat sat on the mat"}


def test_layer_ig_is_complete(mgr):  # noqa: F811
    run = mgr.create_from_spec(TINY_LM, None, TEXT)["id"]
    res = mgr.attribution(run, "ig", None, 64)
    tok = res["tokens"]
    assert res["input"] is None and res["input_note"] is None and len(tok["scores"]) == len(TEXT["text"])
    assert tok["signed"] and tok["layer"] == "transformer.wte" and tok["baseline_id"] == 0
    assert tok["method_label"] == "Layer IG on transformer.wte · baseline token 0"
    model, x = (live := mgr.live(run)).loaded.model, live.inputs
    with torch.no_grad():
        f = model(x).logits[0, -1, res["target"]] - model(torch.zeros_like(x)).logits[0, -1, res["target"]]
    assert tok["total"] == pytest.approx(sum(tok["scores"]))
    assert abs(tok["total"] - float(f)) <= tok["convergence_delta"] + 1e-4 and tok["convergence_delta"] < 1e-2


def test_gxa_and_saliency(mgr):  # noqa: F811
    run = mgr.create_from_spec(TINY_LM, None, TEXT)["id"]
    gxa = mgr.attribution(run, "gxa", None, 8)["tokens"]
    sal = mgr.attribution(run, "saliency", None, 8)["tokens"]
    assert gxa["signed"] and gxa["convergence_delta"] is None and gxa["baseline_id"] is None
    assert gxa["method_label"] == "Gradient x embedding on transformer.wte"
    assert not sal["signed"] and all(s >= 0 for s in sal["scores"]) and any(s > 0 for s in sal["scores"])
    assert sal["method_label"] == "Saliency ‖grad‖ on transformer.wte"
    unit = next(o["id"] for o in mgr.store.ir(run)["ops"] if o.get("module") == "transformer.h.0.ln_2")
    obj = mgr.attribution(run, "ig", None, 8, {"op": unit, "channel": 0})
    assert len(obj["tokens"]["scores"]) == len(TEXT["text"])


def test_stale_cache_agreement_and_layer_message(mgr):  # noqa: F811
    run = mgr.create_from_spec(TINY_LM, None, TEXT)["id"]
    fresh = mgr.attribution(run, "gxa", None, 8)
    old = {k: v for k, v in fresh.items() if k != "tokens"}  # as written before v0.2: no tokens key
    mgr.store.write_json(run, "attribution/gxa_top_8.json", old)
    assert mgr._cached(run, "attribution/gxa_top_8.json") is None
    assert mgr.attribution(run, "gxa", None, 8)["tokens"] == fresh["tokens"]

    ag = mgr.agreement(run, None, 8)
    assert ag["input"]["basis"] == "|attribution| · per token" and ag["input"]["n"] == len(TEXT["text"])
    op = next(o["id"] for o in mgr.store.ir(run)["ops"] if o.get("module") == "transformer.h.0.ln_2")
    job = follow(mgr.jobs, mgr.job(run, {"kind": "layer", "op": op}).id)[-1]
    assert job["status"] == "error" and "token inputs get Layer IG on the embeddings" in job["error"]


def test_embedding_layer_and_baseline():
    from test_errors_m4 import Embed, Small

    assert embedding_layer(Embed())[0] == "emb" and embedding_layer(Small()) is None
    tok = type("T", (), {"pad_token_id": None, "eos_token_id": 50256})()
    assert _baseline_id(tok) == 50256 and _baseline_id(None) == 0
    assert _baseline_id(type("P", (), {"pad_token_id": 3})()) == 3
