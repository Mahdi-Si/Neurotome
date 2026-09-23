# Architecture

## Flow of one run

1. **Load.** `models.load_model(spec, weights)` returns the `nn.Module` plus preprocessing and class names.
2. **Input.** `RunManager.build_input` turns an upload (image / `.npy` / `.pt`) or a random spec into a tensor.
3. **Capture.** `capture.capture()` calls `torchlens.trace(model, x, layers_to_save="all")`, drops buffer
   pseudo-ops, gives every op a stable id, and saves each activation to `tensors/<op>.npy` (float32).
   It computes summary stats for every op and writes `ir.json`. For image-like inputs it also writes
   `input_preview.png` (see Run folder).
4. **Serve.** The API reads `ir.json` to build graph views on demand, and memory-maps `.npy` files to build tensor views.
5. **Attribute** (on request). `attribution.attribute()` re-traces with gradients, back-propagates the
   target logit, and returns grad×activation for every op. It also runs a Captum method on the input.
   Results are cached in `attribution/<method>_<target>_<steps>.json`.

## Run folder (`runs/<run_id>/`)

```
meta.json            run summary: model spec, sample, output top-k, preview {width, height} | null
ir.json              the captured graph (below)
inputs.pt            the exact input tensors, so attribution can re-run later
input_preview.png    the input as a picture, <= 224 px; only for (N, 1|3, H, W) inputs with H, W >= 8
tensors/<op>.npy     one activation per op, float32, memory-mapped on read
attribution/*.json   cached attribution results
```

The preview is de-normalised with the model's `(mean, std)` for uploaded images and min-max stretched per
channel for random, tensor and in-process inputs. Runs captured before M2 have no preview and no `meta.preview`;
they still open, and the UI disables the overlay.

## Backend modules (`backend/neurotome/`)

| File | Responsibility |
|---|---|
| `config.py` | Settings from env: `NEUROTOME_RUNS_DIR`, `NEUROTOME_HOST`, `NEUROTOME_PORT`, `NEUROTOME_DEVICE`, `NEUROTOME_MAX_CAPTURE_MB`, `NEUROTOME_MAX_LIVE_MODELS` |
| `models.py` | Model specs (`torchvision:<name>`, `py:<module>:<factory>`), preprocessing, `to_logits()` |
| `capture.py` | **The only TorchLens-aware file.** Trace → IR; per-op gradients for attribution |
| `store.py` | Run folders, JSON read/write (atomic), `.npy` save / memory-mapped load |
| `graph.py` | `graph_view(ir, depth, scores, expand, collapse)` fold algorithm; `search_index(ir)`; `module_tree(ir)`; `attach_scores` |
| `tensors.py` | `tensor_view(array, channel, pixel, norm, cmap, lo, hi)` → `image` / `sequence` / `vector` summaries; `input_preview`; `tensor_diff` |
| `slicing.py` | Regex-only NumPy-style slice grammar (no `eval`): `slice_table` (≤ 4096 values as JSON), `slice_npy` (≤ 2²⁴ values) |
| `attribution.py` | Captum input methods and assembly of the attribution result |
| `runs.py` | `RunManager`: create runs, keep live models, cache attribution, one-job-at-a-time lock |
| `api.py` | FastAPI app factory; serves the built UI from `static/` when present |
| `cli.py` | `neurotome serve`, `neurotome capture` |
| `__init__.py` | `neurotome.open(model, x)` for scripts and notebooks |

## IR (`ir.json`)

```jsonc
{
  "model": {"class_name": "ResNet", "num_params": 11689512, "num_ops": 69, "flops": 3645279184,
            "activation_bytes": 33538752, "truncated": false},
  "ops": [{
    "id": "conv2d_3_8",                    // stable across re-traces of the same model+input
    "type": "conv2d", "func": "conv2d",
    "shape": [1, 64, 56, 56], "dtype": "torch.float32",
    "device": "cuda:0",                    // first output tensor's device, or null (absent before M1)
    "parents": ["relu_2_7"], "children": ["batchnorm_3_9"],
    "modules": ["layer1:1", "layer1.0:1", "layer1.0.conv2:1"],   // module-call stack, outermost first
    "module": "layer1.0.conv2",            // leaf module that produced it, or null
    "num_params": 36864, "flops": 231211008,
    "is_input": false, "is_output": false, "saved": true,
    "stats": {"mean": -0.01, "std": 0.27, "min": -1.04, "max": 0.97, "zero_frac": 0.0,
              "hist": [/* 32 counts */], "hist_range": [-1.04, 0.97], "numel": 200704,
              "nan": 0, "inf": 0}          // moments and hist over finite values; nan/inf absent before M1
  }],
  "modules": {"layer1.0": {"class_name": "BasicBlock", "num_params": 73984, "depth": 2}}
}
```

Module *calls* (`layer1.0:1`) rather than addresses are used for grouping, so a module called twice
(a shared ReLU, a recurrent cell) produces separate groups.

New fields are optional: runs captured before M1 still open, and readers use `.get()`, so a missing
`device` or `nan`/`inf` shows up as `null` in the API (`device`, `nonfinite`).

## Graph view algorithm (`graph.py`)

Given `depth` *d* and the per-call overrides `expand` / `collapse` (sets of module calls; unknown calls
are ignored):

1. A call at index *i* of a stack is *collapsed* when it is in `collapse`, or when `i ≥ d` and it is not
   in `expand` (`collapse` wins over `expand`). For each op with module stack *S* (empty for
   input/output), let *i\** be the first index whose call is collapsed and contains more than one op. If it
   exists, the op folds into the **module node** `m:S[i*]` with containers `S[:i*]`. Otherwise it stays an
   **op node**. With no overrides this is the plain depth rule.
2. Containers of a node are the module calls above it that hold more than one op. A container becomes
   a **group** (dashed box) only if it holds two or more visible nodes, which avoids a box around every conv.
3. The node's representative tensor is its last member op that has an edge leaving the node.
4. Edges connect the visible nodes of each op's parents; self-loops are dropped and duplicates merged.
   An edge's `shape`, `dtype` and `device` come from the parent op of the first op pair that creates it,
   and `target_dtype` / `target_device` from the consuming op. A node's `dtype` and `device` are its
   representative op's, and `nonfinite` is Σ `nan + inf` over all member ops (`null` when not recorded).
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

Image maps and dots are scaled on the server. `norm=channel` (default) scales each pooled grid by its own
`[min, max]`; `norm=global` uses one range for every map, `[lo, hi]` when both are given (the compare view's shared
scale) else the finite `[min, max]` of the whole tensor. `cmap=seq` maps the range to [0, 1]; `cmap=div` divides by
`max(|lo|, |hi|)` into [-1, 1], so 0 stays 0. The response echoes `norm`, `cmap`, `scale` (the global range or
`null`), each map's `range` and `dots_range`. `peak` is the argmax of the selected channel; the inspected position
is `vector.pixel` / `profile.row`.

**Slices** (`slicing.py`): `[0, 12, 20:30, :]`, `[..., 0]`, `[-1]`, `::-2`. The JSON table is truncated from the last
axis backwards to 4096 values and carries `key` (canonical), `shape`, `full_shape`, `truncated`, `axes`, `indices`
and flat C-order `values` (6 significant digits; non-finite values as `"nan"` / `"inf"` / `"-inf"`).

**Diff** (`tensor_diff`): statistics of b − a over positions where both are finite: `stats`, a 24-bin `hist`,
`max_abs`, `mean_abs`, `cosine`, `rel_l2`; `diff` is `null` when the shapes differ.

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
| GET | `/api/runs/{id}/graph?depth=2[&expand=a:1,b:1&collapse=c:1][&method=ig&target=207&steps=32]` | graph view; `expand` / `collapse` are comma-separated module calls; with `method`, scores, shares and path attached |
| GET | `/api/runs/{id}/index` | search index: every op, then every multi-op module call, as `{id, kind, label, path, type, shape, calls}` |
| GET | `/api/runs/{id}/ops/{op}` | one IR op |
| GET | `/api/runs/{id}/ops/{op}/tensor?channel=&py=&px=&norm=channel\|global&cmap=seq\|div&lo=&hi=` | tensor view; bad `norm` / `cmap` → 422 |
| GET | `/api/runs/{id}/ops/{op}/slice?s=<expr>` | slice as a JSON table (≤ 4096 values); bad expression → 400 with a readable `detail` |
| GET | `/api/runs/{id}/ops/{op}/slice.npy?s=<expr>` | the whole slice as a float32 `.npy` download; 400 bad expression, 413 over 2²⁴ values |
| GET | `/api/runs/{id}/diff?a=<op>&b=<op>` | `{a, b, shape_a, shape_b, same_shape, diff}`; 404 when either tensor is missing |
| GET | `/api/runs/{id}/input.png` | input preview (PNG); 404 for runs without one |
| POST | `/api/runs/{id}/attribution` | `{method: ig\|saliency\|gxa, target, steps}` → cached result |

Interactive docs live at `http://127.0.0.1:7070/docs` while the server runs.

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `lib/api.ts`, `lib/types.ts` | Typed client for the API above |
| `lib/router.ts` | Hash routes `#/runs/<id>/<graph\|tensors\|trace>?node=…` (shareable deep links; the graph view also keeps `depth`, `expand`, `collapse` and `color`) |
| `lib/layout.ts` | ELK layered layout with nested groups. It runs in an ELK Web Worker (lazy-loaded; falls back to the bundled ELK where `Worker` is missing) behind a content-keyed cache, so pan, zoom, attribution and colour changes never re-lay out |
| `lib/viewport.ts` | Pure viewport maths: visible-node and edge culling, node/edge hit tests, keyboard `step`, colour `mix` |
| `lib/graphState.ts` | Pure graph-workspace state: fold (`expand`/`collapse`) URL params, search matching, colour modes |
| `lib/tensorState.ts` | Pure tensor-inspector maths: pixel ↔ pooled-cell mapping (mirrors `_pool1d`), `unscale` back to real values, URL params |
| `lib/exportPng.ts` | PNG export of any chart (SVG, canvas, layered overlay, channel tiles) |
| `components/GraphCanvas.tsx` | Pan/zoom canvas. Edges and group boxes are drawn on a `<canvas>`; nodes are virtualised `<button>`s (only those near the viewport, and drawn on the canvas when zoomed far out); `plain` and `flow` modes |
| `components/Minimap.tsx` | Overview of the whole graph with the viewport rectangle; drag to pan |
| `components/GraphSearch.tsx` | Search box over the `/index` entries (loaded once by GraphView, which also uses them so a ModuleTree pick folds and selects the module itself); selecting a result reveals and centres it |
| `components/charts.tsx` | Histogram, DotMatrix, LinePlot, DivergingBars, Scatter, HeatCanvas; every chart has a hover readout and a PNG export |
| `components/InputOverlay.tsx` | A channel or attribution map drawn over `input.png`, with opacity control |
| `components/ValuesTable.tsx` | Slice expression box, the capped values table and the `.npy` download |
| `components/ComparePanel.tsx` | Compare two ops on a shared scale, with the diff statistics |
| `components/panels.tsx` | Module tree, node inspector |
| `views/*` | Home, Graph, Tensors, Trace |
| `styles/tokens.css` | Design tokens (mirrors `design/tokens.json`) |

Tensor view URL: `#/runs/<id>/tensors?node=…&c=…&py=…&px=…&norm=global&cmap=div&cmp=…` (defaults omitted; `cmp` turns on
compare mode). Overlay on/off and opacity are view state, not URL state.

Dev: Vite on :5173 proxies `/api` to :7070. Build: output goes to `backend/neurotome/static/`, so the Python package serves the UI;
`index.html` is served with `Cache-Control: no-cache` so a rebuilt UI never points at deleted hashed assets.

## Extension points

- **New capture backend** (NNsight, `torch.export`, forward hooks): implement `capture()` and
  `op_gradients()` returning the same IR and score dicts. Nothing downstream changes.
- **New tensor view kind**: add a branch in `tensor_view()` returning `{kind: "...", ...}`, add the type
  in `lib/types.ts`, and add a body component in `views/TensorView.tsx`. The values table, `.npy` export and
  diff work on any array, so a new kind gets them for free; `norm`/`cmap` apply to the image kind only.
- **New attribution method**: add it to `METHODS` and `input_attribution()`; for per-node methods,
  return scores keyed by op id and the graph overlay works unchanged.
- **New node colour mode**: add the id to `ColorMode` in `lib/types.ts`, a branch in `nodeTones()` in
  `lib/graphState.ts` returning `{v, flag?, readout}` per node plus a legend, and a button in
  `views/GraphView.tsx`. GraphCanvas and the minimap draw any tone without changes. If the mode needs
  data the view lacks, add an optional field to the node in `graph_view` (read with `.get`, `null` for old runs).
