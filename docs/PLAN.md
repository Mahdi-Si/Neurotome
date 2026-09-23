# Neurotome: MVP, roadmap and implementation plan

*Version 0.1 · 22 September 2026*

Neurotome is a local web workbench for looking inside one PyTorch model on one input. You capture
a single forward pass, browse the full computation graph and open any tensor as maps, signals or
projections. You can then trace which parts of the model, and which parts of the input, drove a
given output.

This repository is the **v0.1 starter**. Every piece of the pipeline already works end to end,
from capture to graph, tensor and attribution views, and the UI follows the look of the design
mockups. The plan below takes that starter to a releasable MVP and then beyond.

---

## 1. Feasibility verdict

**Yes, this is buildable, and most of the hard parts already exist as libraries.** What's missing
is the product that ties them together behind a good UI.

| Need | Solved by | Status in this repo |
|---|---|---|
| Record every op, not only module outputs, including dynamic control flow | **TorchLens** (`tl.trace`) | Used through one adapter file |
| Gradients for every op in one backward pass | TorchLens `save_grads` + `log_backward` | Used for node and edge attribution |
| Input attribution (IG, saliency, grad×input) | **Captum** | Used |
| Hierarchical graph layout | **ELK** (elkjs) | Used |
| Interventions and patching | TorchLens interventions / **NNsight** | Roadmap (v0.3) |
| LLM-scale remote execution | NNsight + NDIF | Roadmap (v0.2 / v0.4) |

## 2. Landscape and positioning

| Project | What it does well | Why it's not enough on its own |
|---|---|---|
| TorchLens | Exhaustive capture of every op and gradient; static Graphviz output | No interactive UI; outputs are notebook objects |
| Google Model Explorer | Very large graphs, per-node data overlays | Needs `torch.export`, so no data-dependent control flow; no activation viewers |
| TorchExplorer | Module-level inputs/outputs and gradient histograms in a web UI | Module level only; built for training monitoring, not dissecting one sample |
| Comgra | GUI over captured activations | Needs you to instrument code; limited graph |
| Zetane Viewer | 3D tensor viewer | ONNX-based; custom inputs behind a paid tier |
| Treescope | Faceted tensor rendering in notebooks | Notebook pretty-printer, not a workbench |
| Captum Insights | Attribution UI | **Retired in Captum 0.9**, so Captum has no official UI now |
| NNsight / NDIF workbench | Interventions on LLMs, remote big models | LLM-focused; not a general graph and tensor browser |
| circuit-tracer + Neuronpedia | Attribution graphs for LLM features | Needs pre-trained transcoders; only specific LLMs |

**Where Neurotome fits.** It's the only tool in this list that combines (a) an op-level graph for
*any* PyTorch model, (b) tensor viewers chosen by shape, and (c) attribution laid over the same
graph, in one local UI with a deliberate visual language.

## 3. Users and jobs

| # | Job | Example |
|---|---|---|
| J1 | Understand an unfamiliar architecture | "What actually happens inside a ConvNeXt block?" |
| J2 | See what a layer does to *this* sample | "Which channels in `layer3` fire on the dog's face?" |
| J3 | Debug numerics | Dead ReLUs, saturation, NaNs, exploding activations after a code change |
| J4 | Explain a prediction | "Which pixels, channels and paths produced logit 207?" |
| J5 | Compare (post-MVP) | Two samples, two checkpoints, float vs quantized |

Primary persona: an ML engineer or researcher who runs PyTorch locally and uses notebooks, but
wants a spatial, clickable view instead of print statements.

## 4. Product principles

1. **One sample, full trace.** The graph shown is the one this input actually took, not an idealized architecture.
2. **The browser never gets whole tensors.** The server sends summaries: pooled maps, histograms, projections.
3. **Swappable capture.** Only `capture.py` talks to TorchLens. The rest reads Neurotome's own IR.
4. **Honest attribution.** Every number says which method produced it. Approximations are labelled as approximations.
5. **Local-first.** Runs live in a folder on your disk. The server binds to localhost.
6. **Quiet visual language.** Thin hairlines, mono labels, one accent. Color carries meaning only; see [DESIGN.md](DESIGN.md).

## 5. MVP definition

### In scope

| Area | MVP capability | Acceptance check |
|---|---|---|
| Load | torchvision models by name (random or pretrained weights); your own model via `py:module:factory`; `neurotome.open(model, x)` from Python | ResNet-18/50, MobileNetV3, EfficientNet-B0, ConvNeXt-T and ViT-B/16 all capture without errors |
| Sample | Upload an image, `.npy` or `.pt`; or use a random tensor | Upload flows through preprocessing into a run |
| Graph | Op-level graph; collapse by module depth; module groups; module tree; inspector with shape, params, FLOPs and activation histogram | 500-node view pans and zooms smoothly; click a module to select its node |
| Tensors | Channel grid, dot matrix, row profile, channel vector, PCA, histogram (images); heatmap, token norms and PCA (sequences); signal plus top-k (vectors) | Every saved op opens in one of the three kinds |
| Attribution | IG / saliency / grad×input on the input; grad×activation on every node; share at merges; dominant path; top channels | IG completeness Δ is shown; results are cached per (method, target, steps) |
| Ops | CLI `serve` / `capture`; pip-installable package with the UI built in | `pip install .` then `neurotome serve` works on a clean machine |

### Out of scope for MVP

LLM-specific views (attention patterns, logit lens), interventions and patching, multi-sample
comparison, training-time monitoring, multi-user hosting, JAX and TensorFlow.

### Success criteria

- Time from `pip install` to first graph: under 5 minutes.
- ResNet-50 capture on CPU in under 10 s; tensor views in under 300 ms; IG with 32 steps in under 30 s on CPU.
- 5 external users use it on their own model without code changes to Neurotome.

**Measured** on 23 September 2026 with `python scripts/bench.py` (CPU, random weights, temp runs dir):

| Metric | Value | Budget | Pass |
|---|---|---|---|
| `capture_resnet50_s` (ResNet-50 capture, model build included) | 2.56 s | < 10 s | yes |
| `tensor_view_p95_ms` (p95 of `GET …/tensor` over all 177 saved ops) | 165 ms | < 300 ms | yes |
| `ig32_s` (Integrated Gradients, 32 steps) | 4.19 s | < 30 s | yes |

Machine: 13th Gen Intel Core i7-13800H, 14 torch threads (20 logical CPUs), torch 2.14.0, Python 3.12.1, Windows 11.
Four runs that day ranged 2.2–3.8 s, 165–206 ms and 4.0–4.2 s (other work shared the machine).

## 6. Architecture (summary)

```
 Browser (React + Vite)                        Python (FastAPI)                       Disk
 ┌──────────────────────────┐   JSON/HTTP     ┌───────────────────────────────┐     runs/run_0001/
 │ Home · Graph · Tensors · │ ──────────────▶ │ api.py      routes            │     ├ meta.json
 │ Trace views              │                 │ runs.py     RunManager        │     ├ ir.json
 │ GraphCanvas (ELK layout) │ ◀────────────── │ capture.py  TorchLens → IR ◀──┼──┐  ├ inputs.pt
 │ charts (SVG / canvas)    │  summaries only │ graph.py    collapse, scores  │  │  ├ tensors/<op>.npy
 └──────────────────────────┘                 │ tensors.py  shape → view      │  │  └ attribution/*.json
                                              │ attribution.py Captum + g×a   │  │
                                              │ store.py    run folders ──────┼──┘
                                              └───────────────────────────────┘
```

Full details (IR schema, API, algorithms, extension points) are in [ARCHITECTURE.md](ARCHITECTURE.md).

## 7. Implementation plan

Estimates assume one engineer working full time. `[x]` = done in this starter.

### M0 · Foundations (done)

- [x] Monorepo layout: `backend/` (Python package), `frontend/` (Vite + React + TS), `docs/`, `design/`, `examples/`
- [x] Capture adapter over TorchLens 2.34 producing Neurotome's IR (`capture.py`)
- [x] Run store: one folder per run, memory-mapped `.npy` activations, JSON metadata
- [x] Graph view with depth collapsing, module groups and edges between visible nodes (`graph.py`)
- [x] Tensor views chosen by shape: image, sequence, vector (`tensors.py`)
- [x] Attribution: Captum input methods plus per-op grad×activation, merge shares, dominant path
- [x] FastAPI routes, CLI (`neurotome serve`, `neurotome capture`), `neurotome.open()` from Python
- [x] Frontend: 4 views, ELK layout, pan/zoom canvas, charts, design tokens, bundled fonts
- [x] Tests: backend end-to-end (tiny model + ResNet-18), frontend unit tests, CI workflow
- [x] Smoke-tested end to end (capture → every graph depth → every tensor view → attribution) on ResNet-18, ResNet-50, MobileNetV3-S, EfficientNet-B0, VGG-16, ConvNeXt-T and ViT-B/16; capture takes 1–4 s on CPU

### M1 · Graph workspace (done, spec: `docs/specs/m1.md`)

- [x] Expand or collapse *individual* modules (per-node `+` / per-group `−` toggle, `expand=`/`collapse=` in the URL), not only a global depth
- [x] Move layout into a Web Worker (`elk-worker.min.js`, bundled fallback) and cache positions by graph structure, i.e. per (run, depth, expansion state)
- [x] Minimap plus search: jump to a node by name, type or shape; a hit folded inside a collapsed module is revealed
- [x] Keyboard navigation: arrows follow edges, `Enter` opens the tensor, `Escape` clears search, `+`/`-` fold, `/` searches
- [x] Node color modes: attribution, activation norm (RMS), sparsity (zero %), NaN/Inf flag, with a legend
- [x] Hover an edge to see the tensor shape it carries, and dtype/device changes
- [x] Performance budget: 2k nodes (virtualised DOM nodes, canvas edges, no re-layout on pan/zoom; `lib/perf.test.ts` benchmark)

### M2 · Tensor inspector (done, spec: `docs/specs/m2.md`)

- [x] Click a pixel in the dot matrix (or the selected channel's tile) to set the inspected position; the channel vector and profile row follow, `py`/`px` live in the URL, a crosshair marks the pixel
- [x] Global vs per-channel normalization toggle (server-side, `norm=`); sequential and diverging colormaps (`cmap=`), both in the URL
- [x] Input overlay: draw a channel map or the input attribution over `input_preview.png` (written at capture, ≤224 px; old runs disable the toggle)
- [x] Raw values table with NumPy-style slice syntax (`[0, 12, 20:30, :]`): regex-only parser, 4096-value cap with a truncation note, paged 2-D tables
- [x] Compare two nodes side by side (default: the node's parent) with a shared scale when shapes match, plus a Δ summary (stats, histogram, cosine, max |Δ|)
- [x] Export: PNG of any chart (client-side), `.npy` of any slice (capped at 2²⁴ values, 413 beyond)
- [x] M1 carry-over: picking an open module in the ModuleTree folds and selects the module itself, like search

### M3 · Attribution and trace (≈2 weeks)

- [x] Per-node **layer** methods from Captum: LayerIntegratedGradients, LayerConductance, and NeuronConductance for a chosen channel
- [x] Proper merge attribution: split flow at `add`/`cat` by each parent's contribution to the merge output, not the parents' own totals
- [x] "Trace from here": attribute a chosen internal channel, not only an output logit
- [x] Method agreement panel: top-k overlap between methods
- [x] Async jobs with progress for IG and patching (see M4)
- [x] Activation patching v1: replace one node's output with a baseline and re-run (via TorchLens interventions)
- [x] M3 notes: neuron conductance is computed directly (Σ over positions; Captum's `NeuronConductance` refuses aggregated neurons) and matches Captum per neuron; random-init models are seeded on load so a spec reloads to the captured weights; patch deltas skip ops whose tensor a later in-place op (`+=`, `ReLU(inplace)`) overwrote during replay

### M4 · Hardening and release (≈1 week)

- [x] Background job queue for captures: `POST /api/runs` returns 202 and a `capture` job (Load model, Build input, Trace, Save tensors i/N, Summarise) followed over SSE; Home shows the progress bar and Cancel, then opens the run. The CLI and `neurotome.open()` stay synchronous
- [x] Capture budget UI and selective capture: "never cross" budget rule, Truncated / Selective banner in the graph and tensor views, dotted `∅` nodes without a tensor (stats still computed for every op), `modules` and a per-run budget in the form, the API and `neurotome capture --modules --max-capture-mb`
- [x] Error states: `errors.describe()` maps trace failures (422), non-tensor or integer outputs, integer inputs (input attribution skipped with a note; layer methods refuse), missing tensors (404 with the reason), models that can no longer be loaded (409), no-gradient models (400) and CUDA out-of-memory (413, suggests CPU or a smaller budget); the UI shows the message, never a traceback
- [x] Dark theme (token-driven, follows `prefers-color-scheme`, toggle in the top bar stored in `localStorage`; canvases, charts and PNG exports read tokens at draw time; contrast recorded in DESIGN.md), plus the reduced-motion and forced-colours passes
- [x] Build: the wheel and sdist carry the UI (`scripts/check_wheel.py --install` checks a clean venv); PyPI release workflow with trusted publishing behind the `pypi` environment and a versioned mkdocs-material + mike docs site are **prepared, not published** (see RELEASE.md)
- [x] Browser end-to-end tests (Playwright, `npm run e2e`, CI job) over the four views; `scripts/bench.py` for the success criteria (§5)
- [x] Type-check hygiene: `pyrightconfig.json`, `pyright backend/neurotome` at 0 errors in CI, LF line endings

**MVP total: about 6–7 weeks** after this starter.

## 8. Roadmap after MVP

| Release | Theme | Highlights |
|---|---|---|
| v0.2 | Sequences and LLMs | Token input box; Hugging Face models; attention-pattern viewer; logit lens; LayerIntegratedGradients on embeddings; NNsight capture backend for large models |
| v0.3 | Interventions | Zero/mean/resample ablation, activation patching and steering from the UI; before/after diff of every downstream node |
| v0.4 | Scale | Streaming capture to chunked storage (zarr); selective capture during the trace (M4 selects what reaches disk); remote GPU server; optional NDIF backend for very large models |
| v0.5 | Comparison | Two samples or two checkpoints side by side; dataset-level channel statistics; float vs quantized error overlay |
| v1.0 | Ecosystem | Jupyter widget; plugin API for custom tensor viewers and attribution methods; static shareable export; circuit-tracer import |

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| TorchLens API churn (it changed between 2.x minors) | High | Pinned to `>=2.34,<2.35`; all calls live in `capture.py`; tests catch breakage; a forward-hook fallback adapter is possible (module outputs only) |
| Memory blow-up on big models | High | Capture budget (already enforced); selective capture; float16 storage option; streaming to disk (v0.4) |
| Unreadable graphs for large models | Medium | Depth collapsing (done), per-module expansion, search, minimap, worker layout |
| Misleading attribution | Medium | Always label the method; show IG completeness Δ; method agreement panel; docs on caveats |
| Arbitrary code execution from `py:` specs and pickles | Medium | Localhost-only by default; document it; safetensors-only mode for a hosted version |
| ELK slow on huge graphs | Low–Medium | Web Worker, cached layouts, collapse by default above N nodes |

## 10. Open decisions

1. **Name and license.** Resolved: the name is **Neurotome** and the license is Apache-2.0 (matches TorchLens and Captum).
2. **Hosted vs local-only.** The MVP is local. A hosted version needs sandboxing and auth; decide after v0.3.
3. **Second capture backend.** NNsight for LLMs (v0.2) or `torch.export` for static graphs. Recommendation: NNsight, since it serves the LLM roadmap.
4. **Dark mode in MVP or later?** The tokens are ready for it; charts need re-validation.

## 11. Getting started

See [Getting started](index.md) for install, first run, CLI flags and configuration; the repository README has the development setup (`uv` or `make`) and the test, e2e, wheel, docs and benchmark commands.
