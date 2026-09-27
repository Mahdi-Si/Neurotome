"""Review regressions: capture.py, capture_nnsight.py, interventions.py (offline, tiny models)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from neurotome import capture
from neurotome.interventions import generate
from neurotome.models import to_logits


class MaskedAttention(nn.Module):
    """sdpa with a tensor kwarg (attn_mask) computed from the input: a merge with four visible parents."""

    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.m = (nn.Linear(4, n) for n in (4, 4, 4, 3))

    def forward(self, x):
        y = F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x), attn_mask=self.m(x) * 3)
        return y.sum(1)


def test_merge_includes_tensor_kwargs():
    """The VJP re-call passes tensor kwargs: without attn_mask the q / k / v contributions were wrong."""
    torch.manual_seed(0)
    model, x = MaskedAttention().eval(), torch.randn(1, 3, 4)
    merges = capture.op_gradients(model, x, 0, to_logits)["merges"]
    (got,) = [m for op, m in merges.items() if op.startswith("scaleddotproductattention")]
    with torch.no_grad():
        q, k, v, mask = model.q(x), model.k(x), model.v(x), model.m(x) * 3
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, mask)]
    F.scaled_dot_product_attention(*leaves[:3], attn_mask=leaves[3]).sum(1)[0, 0].backward()
    want = [float((t.grad * t).sum()) for t in leaves]
    assert len(got) == 4 and list(got.values()) == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_half_gradient_scores_do_not_overflow():
    """grad × act of a float16 model is taken in float32: 300 · 300 overflowed, inf − inf gave NaN."""
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2) * 300)
        model[1].weight.copy_(torch.tensor([[300.0, -300.0]]))
    res = capture.op_gradients(model.half(), torch.ones(1, 2, dtype=torch.half), 0, to_logits)
    assert res["scores"]["linear_1_1"] == 0.0


def test_generate_bad_site_leaks_no_hook():
    """A site that is not a submodule fails before any hook is registered on the live model."""
    lm = nn.Sequential(nn.Embedding(10, 6), nn.Linear(6, 10)).eval()
    tok = type("Tok", (), {"decode": lambda self, ids: "x"})()
    hooks = [(("1", 1), torch.zeros_like), (("nope", 1), torch.zeros_like)]
    with pytest.raises(AttributeError):
        generate(lm, torch.tensor([[1, 2]]), hooks, 1, tok)
    assert not lm[1]._forward_hooks and not lm._forward_pre_hooks


def test_half_gradient_scores_nnsight():
    pytest.importorskip("nnsight")
    from neurotome import capture_nnsight

    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2) * 300)
        model[1].weight.copy_(torch.tensor([[300.0, -300.0]]))
    res = capture_nnsight.op_gradients(model.half(), torch.ones(1, 2, dtype=torch.half), 0, to_logits)
    assert res["scores"]["0@1"] == 0.0


def test_nnsight_edit_of_a_root_leaf():
    """A model that is itself a leaf (op ``@1``): the edit reaches the output and the output row sees it."""
    pytest.importorskip("nnsight")
    from neurotome import capture_nnsight

    torch.manual_seed(0)
    model, x = nn.Linear(4, 3).eval(), torch.randn(1, 4)
    edits = [("@1", torch.zeros_like)]
    out = capture_nnsight.intervene(model, x, edits, ["@1", "output_1"], keep=["output_1"])
    assert not out["logits_after"].any() and not out["after"]["output_1"].any()
    assert out["nodes"]["output_1"]["l2"] == pytest.approx(float(model(x).norm()), rel=1e-5)


def test_capture_under_no_grad_keeps_op_ids(tmp_path):
    """A no_grad caller recorded one fused transformerencoderlayerfwd op; intervene / op_gradients (grad on)
    see the unfused ops, so the IR ids must come from a grad-enabled trace too."""
    from neurotome.store import RunStore

    store = RunStore(tmp_path / "runs")
    torch.manual_seed(0)
    layer, x = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True).eval(), torch.randn(1, 5, 8)
    ids = []
    for run, grad in (("run_0001", True), ("run_0002", False)):
        store.create(run)
        with torch.set_grad_enabled(grad):
            ids.append([o["id"] for o in capture.capture(layer, x, store, run)["ops"]])
    assert ids[0] == ids[1] and len(ids[0]) > 3
