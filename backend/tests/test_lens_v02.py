"""v0.2 logit lens: rows per block, the last row equals the model's own softmax, ranks, caching, errors."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient
from test_hf_v02 import TINY_LM, mgr, tiny_gpt2  # noqa: F401 - mgr is a fixture
from test_jobs_m3 import follow
from test_pipeline import Tiny

from neurotome.api import create_app
from neurotome.lens import lens_parts, logit_lens, unsupported

transformers = pytest.importorskip("transformers")


def tiny_llama():
    cfg = transformers.LlamaConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                                   num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                                   max_position_embeddings=128)  # fmt: skip
    return transformers.LlamaForCausalLM(cfg)


@pytest.mark.parametrize("build, blocks, norm", [
    (lambda: tiny_gpt2(vocab=64, positions=128), "transformer.h", "transformer.ln_f"),
    (tiny_llama, "model.layers", "model.norm"),
])  # fmt: skip
@pytest.mark.parametrize("T", [7, 70])
def test_lens_rows_match_the_model(build, blocks, norm, T):
    torch.manual_seed(0)
    model = build().eval()
    model.config.use_cache = False
    x = torch.randint(0, 64, (1, T))
    res = logit_lens(model, x, None, None)
    pos0 = max(0, T - 64)
    assert res["layers"] == ["embed", f"{blocks}.0", f"{blocks}.1"] and res["pos0"] == pos0 and res["k"] == 5
    assert res["method_label"] == f"Logit lens · {norm} + lm_head"
    assert res["tokens"] == [str(i) for i in x[0, pos0:].tolist()]
    with torch.no_grad():
        p = torch.softmax(model(x).logits[0, pos0:].float(), -1)
    target = res["target"]["id"]
    assert target == int(p[-1].argmax()) and res["target"]["token"] == str(target)
    last = res["top"][-1]
    assert len(res["top"]) == 3 and len(last) == min(T, 64) and all(len(c) == 5 for c in last)
    for pos, cells in enumerate(last):
        for c in cells:
            assert abs(c["prob"] - float(p[pos, c["id"]])) <= 1e-4
    assert torch.allclose(torch.tensor(res["target_prob"][-1]), p[:, target], atol=1e-4)
    assert res["target_rank"][-1][-1] == 1  # the top prediction is the default target
    for layer, rows in enumerate(res["top"]):
        for pos, cells in enumerate(rows):
            rank = res["target_rank"][layer][pos]
            assert rank >= 1 and (rank == 1 or cells[0]["id"] != target)
    fixed = logit_lens(model, x, 5, None)
    assert fixed["target"]["id"] == 5
    assert fixed["target_prob"][-1][-1] == pytest.approx(float(p[-1, 5]), abs=1e-4)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        logit_lens(model, x, 64, None)


def test_unsupported_models():
    for model in (Tiny(), transformers.GPT2Model(transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2))):
        with pytest.raises(ValueError) as e:
            lens_parts(model)
        assert str(e.value) == unsupported(model) and type(model).__name__ in str(e.value)


def test_lens_job_caches(mgr):  # noqa: F811
    run = mgr.create_from_spec(TINY_LM, None, {"kind": "text", "text": "the cat sat"})["id"]
    first = follow(mgr.jobs, mgr.job(run, {"kind": "lens", "target": None}).id)
    res = first[-1]["result"]
    assert first[-1]["status"] == "done" and res["tokens"] == list("the cat sat")
    assert {m["message"] for m in first} & {"Lens forward", "Project layer 3/3"}
    assert (mgr.store.run_dir(run) / "lens" / "top.json").is_file()
    again = mgr.job(run, {"kind": "lens", "target": None})
    assert again.status == "done" and again.result == res  # born done from the cache
    other = follow(mgr.jobs, mgr.job(run, {"kind": "lens", "target": 3}).id)[-1]["result"]
    assert other["target"] == {"id": 3, "token": "b"}
    assert (mgr.store.run_dir(run) / "lens" / "3.json").is_file()


def test_lens_needs_an_lm(mgr):  # noqa: F811
    c = TestClient(create_app(mgr))
    run = mgr.create_from_objects(Tiny(), torch.randn(1, 3, 16, 16))["id"]
    r = c.post(f"/api/runs/{run}/jobs", json={"kind": "lens", "target": None})
    assert r.status_code == 400 and r.json()["detail"].startswith("Logit lens needs a causal language model")
    with pytest.raises(ValueError, match="causal language model"):
        mgr.lens(run, None)
