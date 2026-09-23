# Architecture

## Flow of one run

0. **Capture job.** `POST /api/runs` validates what needs no model (the spec prefix, the upload's existence; 400
   otherwise) and queues a `capture` job (`RunManager.start_capture`); it answers `202` with the job at once. The
   job reports `Load model`, `Build input`, `Trace`, `Save tensors i/N`, `Summarise` and ends with the run meta as
   its `result`; the browser follows it like any other job (SSE or polling, `Cancel`). A failed or cancelled
   capture leaves no run folder. The CLI and `neurotome.open()` call the same code synchronously.
1. **Load.** `models.load_model(spec, weights)` returns the `nn.Module` plus preprocessing and class names.
   Random-init weights are built under a fixed seed, so reloading a spec later (attribution after a restart or an
   LRU eviction, patching from another run) gives the model the run was captured with.
2. **Input.** `RunManager.build_input` turns an upload (image / `.npy` / `.pt`) or a random spec into a tensor.
   Integer `.npy` arrays (token ids) stay `int64`; everything else becomes float32.
3. **Capture.** `capture.capture()` calls `torchlens.trace(model, x, layers_to_save="all")`, drops buffer
   pseudo-ops, gives every op a stable id, and saves activations to `tensors/<op>.npy` (float32). A trace
   failure becomes a `TraceError` (422) naming the original exception; CUDA OOM passes through (413).
   - **Budget, "never cross".** The per-run budget (`NEUROTOME_MAX_CAPTURE_MB`, or `max_capture_mb` for one
     capture) counts `numel × 4` bytes per op. Input and output ops are always saved and counted first. Walking
     the ops in order, the first non-io op that does not fit and every non-io op after it are skipped
     (`unsaved: "budget"`), so the budget is never crossed and the saved ops are a prefix of the graph.
   - **Selective capture.** `modules` (addresses such as `layer3`, `layer4.1.conv2`) saves only ops whose module
     stack contains a selected module (so `layer3` covers everything under it); the rest are
     `unsaved: "modules"` and use no budget. Unknown addresses fail right after the trace with
     `Unknown module(s) …` and the first few top-level addresses. The trace itself is unchanged: every op stays in
     the IR and the graph.
   - **Stats for every op** with a tensor, saved or not (one CPU copy, then dropped), so colour modes and the
     inspector histogram work on unsaved nodes. Ops without a tensor have `stats: null`, `unsaved: "no_tensor"`.

   `Summarise` then saves the input, the top-k (or `output_error` when the output gives no logits) and, for
   image-like inputs, `input_preview.png`; `ir.json` and `meta.json` are written last (see Run folder).
4. **Serve.** The API reads `ir.json` to build graph views on demand, and memory-maps `.npy` files to build tensor views.
5. **Attribute** (on request, as a job). The objective is a logit or an internal unit (Σ over positions of
   one channel of a module's output). `attribution.attribute()` re-traces with gradients, back-propagates the
   objective, and returns grad×activation for every op plus each parent's grad×input through every merge
   (`merges`). It also runs a Captum method on the input, with a forward hook on the unit's module for a unit
   objective. Layer IG / conductance, neuron conductance, method agreement and activation patching work the same
   way. The Trace view starts them as **jobs** (`POST /api/runs/{id}/jobs`) and follows their progress over SSE;
   results are cached in `attribution/` and `patch/` (below), so a repeated request is answered at once.

## Run folder (`runs/<run_id>/`)

```
meta.json            run summary: model spec + the IR "model" block, sample, output top-k, output_error,
                     preview {width, height} | null
ir.json              the captured graph (below)
inputs.pt            the exact input tensors, so attribution can re-run later
input_preview.png    the input as a picture, <= 224 px; only for (N, 1|3, H, W) inputs with H, W >= 8
tensors/<op>.npy     one activation per saved op, float32, memory-mapped on read
attribution/<method>_<okey>_<steps>.json            input + node attribution (with merges, objective)
attribution/channels_<okey>.json                    per-op per-channel g×a (server-side only)
attribution/<layer_ig|layer_cond>_<okey>_<op>_<steps>.json
attribution/neuron_<okey>_<op>_c<ch>_<steps>.json
attribution/agreement_<okey>_<steps>[_<op>].json
patch/<op>_<baseline>[_<source>]_<target|top>.json
```

`okey` names the objective: `top` (the top prediction), the logit index, or `u_<op>_c<channel>` for a unit.
An attribution file without `merges` (written before M3) is stale and recomputed.

The preview is de-normalised with the model's `(mean, std)` for uploaded images and min-max stretched per
channel for random, tensor and in-process inputs. Runs captured before M2 have no preview and no `meta.preview`;
they still open, and the UI disables the overlay.

`meta.output_error` (since M4) is the `describe()` message of the top-k failure (for example "The model returned
SimpleNamespace, which holds no tensor…"), `null` when the top-k worked; attribution fails with the same text.

## Backend modules (`backend/neurotome/`)

| File | Responsibility |
|---|---|
| `config.py` | Settings from env: `NEUROTOME_RUNS_DIR`, `NEUROTOME_HOST`, `NEUROTOME_PORT`, `NEUROTOME_DEVICE`, `NEUROTOME_MAX_CAPTURE_MB`, `NEUROTOME_MAX_LIVE_MODELS` |
| `errors.py` | `TraceError` (422), `ModelUnavailable` (409), `NO_GRAD`, `oom_message()`, `describe(e) → (status, message)`: the one place exceptions become user-facing text (see Errors) |
| `models.py` | Model specs (`torchvision:<name>`, `py:<module>:<factory>`, random init seeded), `check_spec()`, preprocessing, `to_logits()` (a clear `ValueError` for outputs without float logits) |
| `capture.py` | **The only TorchLens-aware file.** Trace → IR with the never-cross budget and selective capture; per-op gradients (logit or unit objective) and merge VJPs; activation patching by DAG replay |
| `store.py` | Run folders, JSON read/write (atomic), `.npy` save / memory-mapped load |
| `graph.py` | `graph_view(ir, depth, scores, expand, collapse, merges, start)` fold algorithm; `search_index(ir)`; `module_tree(ir)`; `attach_scores`; `module_site(ir, op)` / `module_calls`; `unsaved(ir, op)` (derived for old runs) |
| `tensors.py` | `tensor_view(array, channel, pixel, norm, cmap, lo, hi)` → `image` / `sequence` / `vector` summaries; `input_preview`; `tensor_diff` |
| `slicing.py` | Regex-only NumPy-style slice grammar (no `eval`): `slice_table` (≤ 4096 values as JSON), `slice_npy` (≤ 2²⁴ values) |
| `attribution.py` | Captum input and layer methods, objective hooks, channel conductance, method agreement, assembly of the attribution result |
| `jobs.py` | `JobRunner`: one worker thread, progress, cancellation, `wait()` for SSE; kinds `capture`, `attribution`, `layer`, `neuron`, `agreement`, `patch`; error text through `describe` |
| `runs.py` | `RunManager`: create runs (`start_capture` job or synchronous `create_from_spec` / `create_from_objects`), keep live models, cache attribution / layer / neuron / agreement / patch results, `job()` dispatcher, one-at-a-time lock |
| `api.py` | FastAPI app factory; `ValueError` / `RuntimeError` handlers through `describe()`; serves the built UI from `static/` when present |
| `cli.py` | `neurotome serve`, `neurotome capture [--modules a,b] [--max-capture-mb N]` |
| `__init__.py` | `neurotome.open(model, x)` for scripts and notebooks |

## IR (`ir.json`)

```jsonc
{
  "model": {"class_name": "ResNet", "num_params": 11689512, "num_ops": 69, "flops": 3645279184,
            "activation_bytes": 33538752,     // bytes written to tensors/
            "truncated": false,               // at least one op has unsaved == "budget"
            "budget_bytes": 2147483648,       // the budget this capture used, or null (absent before M4)
            "capture_modules": null,          // ["layer3", "layer4"] for a selective capture (absent before M4)
            "saved_ops": 69},                 // non-io ops with saved == true; compare with num_ops (absent before M4)
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
    "unsaved": null,                       // null iff saved | "budget" | "modules" | "no_tensor" (absent before M4)
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
`device` or `nan`/`inf` shows up as `null` in the API (`device`, `nonfinite`). For runs before M4, graph nodes
derive `unsaved` from `saved`: `null` when saved, else `"budget"` if the run is `truncated`, else `"no_tensor"`.

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
   `unsaved` is the representative op's code (`graph.unsaved`), so the UI can mark nodes without a tensor.
5. With attribution scores: `score` = the representative op's grad×activation, `score_norm` = |score| / max.
   For nodes with more than one incoming edge, an edge's *flow* is Σ |merges[child][parent]| over the op pairs
   that create it, where `merges[c][p]` is parent p's grad×input *through* merge op c (re-calling c's function on
   cloned parents and taking the vector-Jacobian product with c's gradient). When every incoming edge has such a
   pair, `share` = flow / Σ flow and `share_method: merge`. Otherwise (an op that could not be re-called, or a
   module node whose inputs meet inside non-merge ops) the node falls back to parent g×a, `share` = |parent
   score| / Σ, labelled `share_method: parent`. A single incoming edge has `share` 1 and no method, and so has a
   node no flow reaches (score 0 or missing, e.g. after a unit objective): there is nothing to split. The
   **dominant path** is a greedy walk along the highest-share parent from the output, or from the unit's node
   for a unit objective (`start`).

> For linear merges (add, sub, cat, stack) the contributions of an op sum exactly to its own grad×activation, so
> the shares split the merge's flow. A residual's identity branch is credited only with what flows through the
> add, not with its own total (which also counts flow through the conv branch).

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
| GET | `/api/health` | device, torch version, cuda, runs dir, `version` (`neurotome.__version__`), `max_capture_mb` (the server default; `null` when unlimited) |
| GET | `/api/models` | suggested specs, attribution methods |
| GET | `/api/runs` | newest first |
| POST | `/api/uploads` | multipart `file` → `{upload_id}` (≤64 MB) |
| POST | `/api/runs` | `{model, weights, sample: {kind: random\|upload, upload_id, shape, seed}, modules: [address] \| null, max_capture_mb: finite >0 | null}\| null}` → **202** + a `capture` Job (`run_id: null`, `result` = run meta when done); 400 bad spec prefix or missing upload, 422 bad types |
| GET / DELETE | `/api/runs/{id}` | meta / delete run folder |
| GET | `/api/runs/{id}/modules` | module tree for the left rail |
| GET | `/api/runs/{id}/graph?depth=2[&expand=a:1,b:1&collapse=c:1][&method=ig&target=207&steps=32]` | graph view; `expand` / `collapse` are comma-separated module calls; with `method`, scores, shares and path attached |
| GET | `/api/runs/{id}/index` | search index: every op, then every multi-op module call, as `{id, kind, label, path, type, shape, calls}` |
| GET | `/api/runs/{id}/ops/{op}` | one IR op |
| GET | `/api/runs/{id}/ops/{op}/tensor?channel=&py=&px=&norm=channel\|global&cmap=seq\|div&lo=&hi=` | tensor view; bad `norm` / `cmap` → 422; no tensor → 404 (below) |
| GET | `/api/runs/{id}/ops/{op}/slice?s=<expr>` | slice as a JSON table (≤ 4096 values); bad expression → 400 with a readable `detail` |
| GET | `/api/runs/{id}/ops/{op}/slice.npy?s=<expr>` | the whole slice as a float32 `.npy` download; 400 bad expression, 413 over 2²⁴ values |
| GET | `/api/runs/{id}/diff?a=<op>&b=<op>` | `{a, b, shape_a, shape_b, same_shape, diff}`; 404 when either tensor is missing |
| GET | `/api/runs/{id}/input.png` | input preview (PNG); 404 for runs without one |
| GET | `/api/runs/{id}/graph?…&obj_op=<op>&obj_ch=<c>` | with a unit objective (both or neither, else 400): scores, merge shares and a path that starts at the unit |
| POST | `/api/runs/{id}/attribution` | `{method: ig\|saliency\|gxa, target, steps, objective?: {op, channel}}` → cached result (synchronous); 400 on a bad objective |
| POST | `/api/runs/{id}/jobs` | `{kind: attribution\|layer\|neuron\|agreement\|patch, method, target, objective, steps, op, channel, baseline: zeros\|mean\|run, source_run}` → Job; 404 unknown run, 400 missing fields, 422 bad kind / types |
| GET | `/api/jobs/{job_id}` | `{id, kind, run_id, status: queued\|running\|done\|error\|cancelled, progress, message, result, error, created}`; 404 unknown |
| POST | `/api/jobs/{job_id}/cancel` | the job after the cancel request |
| GET | `/api/jobs/{job_id}/events` | `text/event-stream`: `data: <job json>` on every change, `: ping` every 15 s, ends after a terminal state |

`tensor`, `slice`, `slice.npy` and `diff` answer 404 when an op has no tensor on disk, with a detail that says why:
`Op {op} not found in {run_id}` for an op outside the IR, else `No tensor saved for {op}: <reason>` where the reason
is `the capture budget ({mb} MB) was reached before this op; capture again with a larger budget`, `it is outside the
captured modules ({mods}); capture again with its module selected`, or `the op produced no tensor`.

## Errors (`errors.py`)

Every failure reaches the user as one plain message, never a traceback. `describe(e)` maps an exception to
`(status, message)`; the API registers it for `ValueError` and `RuntimeError` (`{"detail": message}`), and the job
runner uses its message as a failed job's `error`. `KeyError` is not mapped globally: routes turn unknown runs, ops
and jobs into their own 404s.

| Exception | Status | Message |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | 413 | `CUDA ran out of memory (…). Capture on the CPU instead (neurotome serve --device cpu or NEUROTOME_DEVICE=cpu), lower the capture budget, or capture fewer modules.` |
| `TraceError` (a `ValueError`) | 422 | `TorchLens could not trace forward(): <type>: <first line>. Check that model(x) runs…` |
| `ModelUnavailable` (a `RuntimeError`) | 409 | an in-process run whose model is gone, or a spec that no longer loads (`The model of run_… (<spec>) can no longer be loaded: …`) |
| `ValueError` | 400 | its text: bad requests, `Cannot load model '<spec>': …`, `Unknown module(s) …`, `The model returned <type>, which holds no tensor…`, integer inputs for layer methods |
| `RuntimeError` naming a missing gradient (`does not require grad`, `…not have been used in the graph`) | 400 | `NO_GRAD`: forward() leaves autograd between the input and the output, or a frozen / parameter-free model has an integer input (`op_gradients` makes float inputs require grad, so frozen models with float inputs still get node gradients) |
| anything else | 500 | `<Type>: <message>` |

An empty message is replaced by the exception's type name. A failed capture job deletes its half-written run folder;
after an OOM it also empties the CUDA cache. Integer inputs (token ids) still get node scores; the attribution
result then carries `input: null` and an `input_note` that says why (`null` whenever an input map was computed).

## Jobs (`jobs.py`)

Captures and attribution, layer, neuron, agreement and patch requests run as jobs on **one** daemon worker thread fed by a
queue, so one job runs at a time. Jobs call `RunManager` methods, which take `RunManager._lock` themselves (the
worker never does; the lock is not re-entrant), so a synchronous capture and a job never share the GPU.
`RunManager.job()` validates the request, answers a cache hit with a job born `done`, and otherwise submits the
work keyed by its cache file, so a second identical request joins the queued or running job (never one already asked to cancel). Work reports
`progress(fraction, message)` (`Node gradients`, `IG batch i/n`, `Layer IG batch i/n`, `Conductance batch i/n`,
`Agreement · <method> (i/N)`, `Clean pass`, `Patched replay`); the fraction is monotone and held ≤ 0.99 until
done. Cancelling a queued job ends it at once; a running job ends at its next progress checkpoint, which raises
`Cancelled`. Capture jobs (`kind: "capture"`, `run_id: null`) report `Load model`, `Build input`, `Trace`,
`Save tensors i/N` (every ~2% of the ops) and `Summarise`; a cancelled capture deletes its run folder. Every change bumps the job's version under one condition variable; the SSE route waits on it
(`data: <job json>`, `: ping` every 15 s, closed after `done` / `error` / `cancelled`). Jobs live in memory; the
newest 100 finished jobs are kept.

**Layer methods** hook a module's output: the op must be the output of a module call (`module_site`), and the
module must be called once (when a block's last op is also a shared ReLU's output, the block itself is used).
With a unit objective the layer must lie strictly before the unit (an ancestor op), else 400 "does not feed the
objective".
Neuron conductance sums one channel's conductance over positions (Captum refuses aggregated neurons).
**Agreement** compares the three input maps and, for an op, g×a / layer IG / layer conductance per channel:
Jaccard and overlap of the top 10% by |value|, and Spearman of average ranks (`null` for a constant vector).

**Patching** (`capture.patch`) re-traces with `intervention_ready=True`, replaces one op's output and replays the
recorded DAG (TorchLens `replace_with`, `engine="replay"`), which updates every downstream op. Baselines: zeros,
the spatial mean (over dims ≥ 2), or the same op's saved tensor from another run of the same model spec and weights. Each
op's delta is ‖after − before‖₂ (and relative to ‖before‖); ops that are not descendants of the patched op are 0.
Replay re-runs in-place functions (`out += identity`, `ReLU(inplace=True)`) on their parents' tensors, so a
descendant whose tensor a later in-place op overwrote no longer holds its own value; it is left out of `deltas`
(the graph shows `‖Δ‖ —`). Logits and every other op match a forward-hook replacement exactly.

Interactive docs live at `http://127.0.0.1:7070/docs` while the server runs.

## Configuration and CLI

`NEUROTOME_RUNS_DIR` (default `./runs`), `NEUROTOME_HOST` / `NEUROTOME_PORT` (`127.0.0.1:7070`),
`NEUROTOME_DEVICE` (`cpu`), `NEUROTOME_MAX_CAPTURE_MB` (the per-run activation budget, default 2048) and
`NEUROTOME_MAX_LIVE_MODELS` (default 2).

- `neurotome serve [--host] [--port] [--runs-dir] [--device] [--reload]`
- `neurotome capture MODEL [--weights] [--image] [--runs-dir] [--modules layer3,layer4] [--max-capture-mb 512]`:
  synchronous; `--modules` is comma-separated (whitespace stripped), `--max-capture-mb` overrides the budget for
  this run.
- `neurotome.open(model, x)` captures in-process (synchronously) and serves the UI.

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `lib/api.ts`, `lib/types.ts` | Typed client for the API above |
| `lib/router.ts` | Hash routes `#/runs/<id>/<graph\|tensors\|trace>?node=…` (shareable deep links; the graph view also keeps `depth`, `expand`, `collapse`, `color` and the patch `patch`/`pb`/`psrc`) |
| `lib/layout.ts` | ELK layered layout with nested groups. It runs in an ELK Web Worker (lazy-loaded; falls back to the bundled ELK where `Worker` is missing) behind a content-keyed cache, so pan, zoom, attribution and colour changes never re-lay out |
| `lib/viewport.ts` | Pure viewport maths: visible-node and edge culling, node/edge hit tests, keyboard `step`, colour `mix` |
| `lib/graphState.ts` | Pure graph-workspace state: fold (`expand`/`collapse`) and patch URL params, search matching, colour modes (incl. `patch`) |
| `lib/traceState.ts` | Trace view URL state (`m`, `t`, `obj`+`ch`, `steps`, `layer`, `lop`, `nc`, `go`), the attribution request, agreement cell tones |
| `lib/jobs.ts` | `followJob` (follow the job a POST starts: SSE or 400 ms polling; abort → server cancel), `runJob` on top of it, and the `useJob` hook (one job slot per panel) |
| `lib/capture.ts` | Pure capture helpers: `hasTensor`, `parseModules`, `parseBudget`, `unsavedText`, `captureNote` (the partial-capture banner text) |
| `lib/theme.ts` | Light / dark theme: `resolveTheme`, `storedFor`, WCAG `contrast`, `readTokens` / `useTokens` (canvases read tokens at draw time), `useTheme` toggle |
| `lib/tensorState.ts` | Pure tensor-inspector maths: pixel ↔ pooled-cell mapping (mirrors `_pool1d`), `unscale` back to real values, URL params |
| `lib/exportPng.ts` | PNG export of any chart (SVG, canvas, layered overlay, channel tiles) |
| `components/GraphCanvas.tsx` | Pan/zoom canvas. Edges and group boxes are drawn on a `<canvas>`; nodes are virtualised `<button>`s (only those near the viewport, and drawn on the canvas when zoomed far out); `plain` and `flow` modes |
| `components/Minimap.tsx` | Overview of the whole graph with the viewport rectangle; drag to pan |
| `components/GraphSearch.tsx` | Search box over the `/index` entries (loaded once by GraphView, which also uses them so a ModuleTree pick folds and selects the module itself); selecting a result reveals and centres it |
| `components/charts.tsx` | Histogram, DotMatrix, LinePlot, DivergingBars, Scatter, HeatCanvas; every chart has a hover readout and a PNG export |
| `components/InputOverlay.tsx` | A channel or attribution map drawn over `input.png`, with opacity control |
| `components/ValuesTable.tsx` | Slice expression box, the capped values table and the `.npy` download |
| `components/ComparePanel.tsx` | Compare two ops on a shared scale, with the diff statistics |
| `components/panels.tsx` | Module tree, node inspector (with `Trace from` channel link, `Layer attribution` link and the patch panel) |
| `components/PatchPanel.tsx` | Patch baseline choice, `Patch & re-run`, Δlogit / Δprob and top-5 after the patch |
| `components/JobProgress.tsx` | Job progress line with `Cancel`; the error text of a failed job |
| `components/CaptureBanner.tsx` | The `Truncated` / `Selective` row under the TopBar for partial captures (Graph and Tensor views) |
| `components/AgreementMatrix.tsx` | n×n method-agreement matrix (Jaccard or Spearman) with a hover readout and PNG export |
| `views/*` | Home, Graph, Tensors, Trace |
| `styles/tokens.css` | Design tokens (mirrors `design/tokens.json`) |

Tensor view URL: `#/runs/<id>/tensors?node=…&c=…&py=…&px=…&norm=global&cmap=div&cmp=…` (defaults omitted; `cmp` turns on
compare mode). Overlay on/off and opacity are view state, not URL state.

Trace view URL: `#/runs/<id>/trace?node=…&m=ig&t=207&obj=<op>&ch=<c>&steps=32&layer=layer_ig&lop=<op>&nc=<c>&go=1`
(defaults omitted). `go=1` runs the attribution, then the layer and neuron jobs, once and drops itself; the
Tensor view and the graph inspector link here with it.

Dev: Vite on :5173 proxies `/api` to :7070. Build: output goes to `backend/neurotome/static/`, so the Python package serves the UI;
`index.html` is served with `Cache-Control: no-cache` so a rebuilt UI never points at deleted hashed assets. Every other path
falls back to `index.html`, except `/api/*` (a JSON 404) and paths that resolve outside `static/` (never served).

## Extension points

- **New capture backend** (NNsight, `torch.export`, forward hooks): implement `capture()`,
  `op_gradients()` (including the `merges` field and the unit objective) and `patch()`, returning the same IR,
  score and delta dicts. Nothing downstream changes.
- **New job kind**: one `RunManager` method that takes `progress` and caches its result, plus a branch in
  `RunManager.job()` (and the kind in `JobRequest`).
- **New tensor view kind**: add a branch in `tensor_view()` returning `{kind: "...", ...}`, add the type
  in `lib/types.ts`, and add a body component in `views/TensorView.tsx`. The values table, `.npy` export and
  diff work on any array, so a new kind gets them for free; `norm`/`cmap` apply to the image kind only.
- **New attribution method**: add it to `METHODS` and `input_attribution()`; for per-node methods,
  return scores keyed by op id and the graph overlay works unchanged.
- **New node colour mode**: add the id to `ColorMode` in `lib/types.ts`, a branch in `nodeTones()` in
  `lib/graphState.ts` returning `{v, flag?, readout}` per node plus a legend, and a button in
  `views/GraphView.tsx`. GraphCanvas and the minimap draw any tone without changes. If the mode needs
  data the view lacks, add an optional field to the node in `graph_view` (read with `.get`, `null` for old runs).
- **New error message**: raise a `ValueError` with the user-facing text (400), or add an exception class to
  `errors.py` and a branch in `describe()`; the API handlers, job errors and UI error lines pick it up unchanged.
- **New colour token**: add it to every block of `styles/tokens.css` (light, both dark blocks, forced colours) and
  to `design/tokens.json`; `lib/theme.test.ts` checks parity and contrast. Canvases read it through `Tokens` /
  `useTokens()` in `lib/theme.ts`, never as a literal.
