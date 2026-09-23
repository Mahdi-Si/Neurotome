# Architecture

## Flow of one run

1. **Load.** `models.load_model(spec, weights)` returns the `nn.Module` plus preprocessing and class names.
2. **Input.** `RunManager.build_input` turns an upload (image / `.npy` / `.pt`) or a random spec into a tensor.
3. **Capture.** `capture.capture()` calls `torchlens.trace(model, x, layers_to_save="all")`, drops buffer
   pseudo-ops, gives every op a stable id, and saves each activation to `tensors/<op>.npy` (float32).
   It computes summary stats for every op and writes `ir.json`.
4. **Serve.** The API reads `ir.json` to build graph views on demand, and memory-maps `.npy` files to build tensor views.
5. **Attribute** (on request). `attribution.attribute()` re-traces with gradients, back-propagates the
   target logit, and returns grad×activation for every op. It also runs a Captum method on the input.
   Results are cached in `attribution/<method>_<target>_<steps>.json`.

## Backend modules (`backend/dissect/`)

| File | Responsibility |
|---|---|
| `config.py` | Settings from env: `DISSECT_RUNS_DIR`, `DISSECT_HOST`, `DISSECT_PORT`, `DISSECT_DEVICE`, `DISSECT_MAX_CAPTURE_MB` |
| `models.py` | Model specs (`torchvision:<name>`, `py:<module>:<factory>`), preprocessing, `to_logits()` |
| `capture.py` | **The only TorchLens-aware file.** Trace → IR; per-op gradients for attribution |
| `store.py` | Run folders, JSON read/write (atomic), `.npy` save / memory-mapped load |
| `graph.py` | `graph_view(ir, depth, scores)` collapse algorithm; `module_tree(ir)`; `attach_scores` |
| `tensors.py` | `tensor_view(array, channel, pixel)` → `image` / `sequence` / `vector` summaries |
| `attribution.py` | Captum input methods and assembly of the attribution result |
| `runs.py` | `RunManager`: create runs, keep live models, cache attribution, one-job-at-a-time lock |
| `api.py` | FastAPI app factory; serves the built UI from `static/` when present |
| `cli.py` | `dissect serve`, `dissect capture` |
| `__init__.py` | `dissect.open(model, x)` for scripts and notebooks |

## IR (`ir.json`)

```jsonc
{
  "model": {"class_name": "ResNet", "num_params": 11689512, "num_ops": 69, "flops": 3645279184,
            "activation_bytes": 33538752, "truncated": false},
  "ops": [{
    "id": "conv2d_3_8",                    // stable across re-traces of the same model+input
    "type": "conv2d", "func": "conv2d",
    "shape": [1, 64, 56, 56], "dtype": "torch.float32",
    "parents": ["relu_2_7"], "children": ["batchnorm_3_9"],
    "modules": ["layer1:1", "layer1.0:1", "layer1.0.conv2:1"],   // module-call stack, outermost first
    "module": "layer1.0.conv2",            // leaf module that produced it, or null
    "num_params": 36864, "flops": 231211008,
    "is_input": false, "is_output": false, "saved": true,
    "stats": {"mean": -0.01, "std": 0.27, "min": -1.04, "max": 0.97, "zero_frac": 0.0,
              "hist": [/* 32 counts */], "hist_range": [-1.04, 0.97], "numel": 200704}
  }],
  "modules": {"layer1.0": {"class_name": "BasicBlock", "num_params": 73984, "depth": 2}}
}
```

Module *calls* (`layer1.0:1`) rather than addresses are used for grouping, so a module called twice
(a shared ReLU, a recurrent cell) produces separate groups.

## Graph view algorithm (`graph.py`)

Given `depth` *d*:

1. For each op with module stack *S*: if `len(S) > d` and module call `S[d]` contains more than one
   op, the op folds into a **module node** `m:S[d]`. Otherwise it stays an **op node**.
2. Containers of a node are the module calls above it that hold more than one op. A container becomes
   a **group** (dashed box) only if it holds two or more visible nodes, which avoids a box around every conv.
3. The node's representative tensor is its last member op that has an edge leaving the node.
4. Edges connect the visible nodes of each op's parents; self-loops are dropped and duplicates merged.
5. With attribution scores: `score` = the representative op's grad×activation, `score_norm` = |score| / max.
   For nodes with more than one parent, each incoming edge gets `share` = |parent score| / Σ. The
   **dominant path** is a greedy walk from the output along the highest-share parent.

> The merge share uses each parent's *own* grad×activation. That's a first-order estimate; M3 replaces it
> with the parent's contribution *to the merge output*.

## Tensor views (`tensors.py`)

After dropping the batch axis:

| Kind | Shape | Payload |
|---|---|---|
| `image` | (C, H, W), H, W > 1 (higher ranks fold into C) | top-32 channels by mean as 24×24 pooled maps; selected channel as 16×16 dots; peak-row profile; channel vector at a pixel; PCA of the H·W positions (≤400 points); histogram |
| `sequence` | (T, D) | heatmap pooled to ≤64×128 (signed); per-token L2 norm; PCA of tokens; histogram |
| `vector` | (D,) | values pooled to ≤512; top-10 by magnitude; histogram |

Everything is pooled on the server, so no response carries more than a few thousand numbers.

## HTTP API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | device, torch version, runs dir |
| GET | `/api/models` | suggested specs, attribution methods |
| GET | `/api/runs` | newest first |
| POST | `/api/uploads` | multipart `file` → `{upload_id}` (≤64 MB) |
| POST | `/api/runs` | `{model, weights, sample: {kind: random\|upload, upload_id, shape, seed}}` → run meta |
| GET / DELETE | `/api/runs/{id}` | meta / delete run folder |
| GET | `/api/runs/{id}/modules` | module tree for the left rail |
| GET | `/api/runs/{id}/graph?depth=2[&method=ig&target=207&steps=32]` | graph view; with `method`, scores, shares and path attached |
| GET | `/api/runs/{id}/ops/{op}` | one IR op |
| GET | `/api/runs/{id}/ops/{op}/tensor?channel=&py=&px=` | tensor view |
| POST | `/api/runs/{id}/attribution` | `{method: ig\|saliency\|gxa, target, steps}` → cached result |

Interactive docs live at `http://127.0.0.1:7070/docs` while the server runs.

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `lib/api.ts`, `lib/types.ts` | Typed client for the API above |
| `lib/router.ts` | Hash routes `#/runs/<id>/<graph\|tensors\|trace>?node=…` (shareable deep links) |
| `lib/layout.ts` | ELK layered layout with nested groups; ELK is lazy-loaded as its own chunk |
| `components/GraphCanvas.tsx` | Pan/zoom canvas; nodes are `<button>`s over an SVG edge layer; `plain` and `flow` modes |
| `components/charts.tsx` | Histogram, DotMatrix, LinePlot, DivergingBars, Scatter, HeatCanvas; every chart has a hover readout |
| `components/panels.tsx` | Module tree, node inspector |
| `views/*` | Home, Graph, Tensors, Trace |
| `styles/tokens.css` | Design tokens (mirrors `design/tokens.json`) |

Dev: Vite on :5173 proxies `/api` to :7070. Build: output goes to `backend/dissect/static/`, so the Python package serves the UI.

## Extension points

- **New capture backend** (NNsight, `torch.export`, forward hooks): implement `capture()` and
  `op_gradients()` returning the same IR and score dicts. Nothing downstream changes.
- **New tensor view kind**: add a branch in `tensor_view()` returning `{kind: "...", ...}`, add the type
  in `lib/types.ts`, and add a body component in `views/TensorView.tsx`.
- **New attribution method**: add it to `METHODS` and `input_attribution()`; for per-node methods,
  return scores keyed by op id and the graph overlay works unchanged.
