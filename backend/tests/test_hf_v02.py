"""v0.2 Hugging Face models and token input: hf: specs, tokenizers, text samples, token meta (all offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from test_jobs_m3 import follow

from neurotome.api import create_app
from neurotome.cli import main
from neurotome.errors import describe
from neurotome.models import HF_MISSING, LoadedModel, Vocab, check_spec, load_model
from neurotome.runs import RunManager
from neurotome.store import RunStore

transformers = pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[2]
TINY_LM = "py:examples.tiny_lm:build"
WORDS = ["[PAD]", "[UNK]", "the", "cat", "sat", "on", "mat", "a", "dog", "ran"]


def tiny_gpt2(vocab: int = 16, positions: int = 64):
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=positions, vocab_size=vocab,
                                  bos_token_id=0, eos_token_id=0)  # fmt: skip
    return transformers.GPT2LMHeadModel(cfg)


def word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    t = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]")


@pytest.fixture()
def hf_dir(tmp_path) -> Path:
    """A tiny GPT-2 and a word-level tokenizer saved like a hub repo, loaded back as hf:<dir>."""
    d = tmp_path / "tiny-gpt2"
    torch.manual_seed(1)
    tiny_gpt2().save_pretrained(d)
    word_tokenizer().save_pretrained(d)
    return d


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO))
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return RunManager(RunStore(tmp_path / "runs"))


def test_spec_prefix_and_missing_extra(monkeypatch):
    assert check_spec("hf:x") == "hf:x"
    with pytest.raises(ValueError, match="'hf:<repo_id>'"):
        check_spec("hub:x")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError) as e:
        load_model("hf:gpt2")
    assert str(e.value) == HF_MISSING and "pip install 'neurotome[hf]'" in HF_MISSING


@pytest.mark.parametrize("weights", [None, "DEFAULT"])
def test_hf_dir_loads(hf_dir, weights):
    loaded = load_model(f"hf:{hf_dir}", weights)
    m = loaded.model
    assert type(m).__name__ == "GPT2LMHeadModel" and loaded.spec == f"hf:{hf_dir}" and loaded.lm
    assert m.config._attn_implementation == "eager" and m.config.use_cache is False and not m.training
    assert isinstance(loaded.categories, Vocab) and len(loaded.categories) == 16
    assert loaded.categories[3] == "cat" and loaded.default_input_shape == [1, 16]
    saved = transformers.GPT2LMHeadModel.from_pretrained(hf_dir)
    same = torch.equal(m.transformer.wte.weight, saved.transformer.wte.weight)
    assert same == (weights == "DEFAULT")
    again = load_model(f"hf:{hf_dir}", weights).model  # random init is seeded: reloads to the same model
    assert torch.equal(m.transformer.h[0].mlp.c_fc.weight, again.transformer.h[0].mlp.c_fc.weight)


def test_hf_image_model(tmp_path):
    from PIL import Image

    d = tmp_path / "tiny-vit"
    cfg = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64,
                                 id2label={0: "cat", 1: "dog"}, label2id={"cat": 0, "dog": 1})  # fmt: skip
    cfg.architectures = ["ViTForImageClassification"]
    transformers.ViTForImageClassification(cfg).save_pretrained(d)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    loaded = load_model(f"hf:{d}")
    assert loaded.tokenizer is None and not loaded.lm and loaded.categories == ["cat", "dog"]
    assert loaded.default_input_shape == [1, 3, 32, 32] and loaded.norm is not None
    assert loaded.preprocess is not None
    assert loaded.preprocess(Image.new("L", (50, 40))).shape == (1, 3, 32, 32)


def test_hf_classifier_and_headless_lm(tmp_path, mgr):
    d = tmp_path / "tiny-cls"  # a text classifier: labels from id2label, never vocabulary tokens
    cfg = transformers.BertConfig(vocab_size=len(WORDS), hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=64, id2label={0: "neg", 1: "pos"},
                                  label2id={"neg": 0, "pos": 1})  # fmt: skip
    transformers.BertForSequenceClassification(cfg).save_pretrained(d)
    word_tokenizer().save_pretrained(d)
    loaded = load_model(f"hf:{d}")
    assert loaded.categories == ["neg", "pos"] and not loaded.lm and loaded.tokenizer is not None
    rnd = mgr.create_from_spec(f"hf:{d}", None, {"kind": "random", "shape": [1, 5]})
    assert rnd["sample"]["name"] == "random tokens (5)"
    assert {o["label"] for o in rnd["output"]} == {"neg", "pos"}
    obj = mgr.create_from_objects(loaded.model, torch.tensor([[2, 3]]), tokenizer=loaded.tokenizer)
    assert "[PAD]" not in {o["label"] for o in obj["output"]}

    d = tmp_path / "no-arch"  # a causal-LM config without architectures loads with its LM head
    m = tiny_gpt2()
    m.config.architectures = None
    m.save_pretrained(d)
    word_tokenizer().save_pretrained(d)
    loaded = load_model(f"hf:{d}")
    assert type(loaded.model).__name__ == "GPT2LMHeadModel" and loaded.lm and len(loaded.categories) == 16


def test_tiny_lm_example(mgr):
    loaded = load_model(TINY_LM)
    assert loaded.lm and loaded.tokenizer is not None and isinstance(loaded.categories, Vocab)
    assert len(loaded.categories) == 33 and loaded.model.config.use_cache is False
    tok = loaded.tokenizer
    assert tok.decode(tok.encode("The cat!")) == "the cat!" and tok.encode("é") == tok.encode("?")


def test_text_sample_meta(mgr, hf_dir):
    meta = mgr.create_from_spec(f"hf:{hf_dir}", None, {"kind": "text", "text": "the cat sat on the mat"})
    s = meta["sample"]
    assert s["kind"] == "text" and s["text"] == s["name"] == "the cat sat on the mat"
    assert s["token_ids"] == [2, 3, 4, 5, 2, 6] and s["tokens"] == ["the", "cat", "sat", "on", "the", "mat"]
    assert meta["categories"] is None and meta["model"]["lm"] is True
    assert meta["model"]["backend"] == "torchlens" and meta["model"]["granularity"] == "op"
    assert all(isinstance(o["label"], str) for o in meta["output"])
    inp = next(o for o in mgr.store.ir(meta["id"])["ops"] if o["is_input"])
    assert inp["dtype"] == "torch.int64" and inp["shape"] == [1, 6]

    rnd = mgr.create_from_spec(f"hf:{hf_dir}", None, {"kind": "random"})
    assert rnd["sample"]["name"] == "random tokens (16)" and len(rnd["sample"]["token_ids"]) == 16
    assert all(0 <= i < 16 for i in rnd["sample"]["token_ids"])


def test_text_sample_errors(mgr, hf_dir):
    spec = f"hf:{hf_dir}"
    for sample, msg in (
        ({"kind": "text", "text": "cat " * 65}, "The text is 65 tokens; this model takes at most 64"),
        ({"kind": "text", "text": "   "}, "The text has no tokens"),
    ):
        with pytest.raises(ValueError) as e:
            mgr.create_from_spec(spec, None, sample)
        assert str(e.value) == msg and describe(e.value)[0] == 400
    with pytest.raises(ValueError, match="This model has no tokenizer; upload a .npy"):
        mgr.create_from_spec("py:test_errors_m4:Small", None, {"kind": "text", "text": "hi"})
    with pytest.raises(ValueError, match="backend must be one of"):
        mgr.create_from_spec(TINY_LM, None, {"kind": "text", "text": "hi"}, backend="bogus")
    assert list(mgr.store.root.glob("run_*")) == []


def test_api_text_run_and_models(mgr):
    c = TestClient(create_app(mgr))
    r = c.post("/api/runs", json={"model": TINY_LM, "sample": {"kind": "text", "text": "hello there"}})
    assert r.status_code == 202
    meta = follow(mgr.jobs, r.json()["id"])[-1]["result"]
    assert c.get(f"/api/runs/{meta['id']}").json()["sample"]["tokens"][:2] == ["h", "e"]
    r = c.post("/api/runs", json={"model": TINY_LM, "sample": {"kind": "text", "text": ""}})
    assert r.status_code == 400 and r.json()["detail"] == "Text sample needs text"
    too_long = {"model": TINY_LM, "sample": {"kind": "text", "text": "x" * 10001}}
    assert c.post("/api/runs", json=too_long).status_code == 422
    assert c.post("/api/runs", json={"model": TINY_LM, "backend": "tvm"}).status_code == 422
    info = c.get("/api/models").json()
    assert info["text"] == ["hf:gpt2", "hf:Qwen/Qwen2.5-0.5B"] and "hf:gpt2" in info["suggested"]
    assert info["backends"]["hf"] is True and isinstance(info["backends"]["nnsight"], bool)
    assert info["nnsight_params"] == 1_000_000_000 and set(info["methods"]) == {"ig", "gxa", "saliency"}


def test_in_process_tokenizer(mgr):
    torch.manual_seed(0)
    model = tiny_gpt2()
    model.config.use_cache = True  # create_from_objects prepares HF models like the loader does
    meta = mgr.create_from_objects(model, torch.tensor([[2, 3, 4]]), tokenizer=word_tokenizer())
    assert meta["sample"]["tokens"] == ["the", "cat", "sat"] and meta["model"]["lm"] is True
    assert model.config.use_cache is False and meta["categories"] is None
    assert sum(o["is_output"] for o in mgr.store.ir(meta["id"])["ops"]) == 1
    plain = mgr.create_from_objects(tiny_gpt2(), torch.tensor([[2, 3]]))
    assert "tokens" not in plain["sample"] and plain["model"]["lm"] is False


def test_cli_text(mgr, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEUROTOME_RUNS_DIR", str(tmp_path / "cli-runs"))
    assert main(["capture", TINY_LM, "--text", "the cat", "--backend", "torchlens"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sample"]["kind"] == "text" and out["sample"]["tokens"] == list("the cat")
    assert out["model"]["backend"] == "torchlens" and out["model"]["lm"] is True
    with pytest.raises(SystemExit):
        main(["capture", TINY_LM, "--text", "a", "--image", "x.png"])


def test_vocab_is_lazy():
    calls = []
    tok = type("T", (), {"decode": lambda self, ids: calls.append(ids) or f"<{ids[0]}>"})()
    v = Vocab(tok, 50_000)
    assert len(v) == 50_000 and v[7] == "<7>" and calls == [[7]]
    with pytest.raises(IndexError):
        v[50_000]
    assert LoadedModel(torch.nn.Linear(1, 1), None, None).lm is False
