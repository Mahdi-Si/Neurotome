# Architecture

## Flow of one run

0. **Capture job.** `POST /api/runs` validates what needs no model (the spec prefix, the upload's existence; 400
   otherwise) and queues a `capture` job (`RunManager.start_capture`); it answers `202` with the job at once. The
   job reports `Load model`, `Build input`, `Trace`, `Save tensors i/N`, `Summarise` and ends with the run meta as
   its `result`; the browser follows it like any other job (SSE or polling, `Cancel`). A failed or cancelled
   capture leaves no run folder. The CLI and `neurotome.open()` call the same code synchronously.
1. **Load.** `models.load_model(spec, weights)` returns the `nn.Module` plus preprocessing and class names
   (`torchvision:<name>`, `hf:<repo_id>` or `py:<module>:<factory>`). Random-init weights are built under a fixed
   seed, so reloading a spec later (attribution after a restart or an LRU eviction, patching from another run)
   gives the model the run was captured with.
   - **Hugging Face** (`hf:gpt2`, `hf:Qwen/Qwen2.5-0.5B`, `hf:google/vit-base-patch16-224`, or a local directory;
     needs the `hf` extra): the class is `getattr(transformers, config.architectures[0])`, else `AutoModel`; random
     init via `_from_config`, `DEFAULT` (any weights value) via `from_pretrained`, float32, **eager attention**
     (so every softmax is an op) and `config.use_cache = False` (else the trace has KV-cache output ops). The same
     preparation applies to any `PreTrainedModel` from a `py:` factory or `neurotome.open`.
   - **Tokenizers**: a model with a tokenizer (`AutoTokenizer`, a `py:` module's `TOKENIZER`, or
     `neurotome.open(..., tokenizer=)`) gets `Vocab` categories: token labels decoded on demand, never a 50k list.
     Neurotome needs only `encode(text)`, `decode(ids)` and optional `pad_token_id` / `eos_token_id`. A causal LM
     (class name ending in `ForCausalLM` / `LMHeadModel`) with a tokenizer is an **LM** (`meta.model.lm`), which
     enables the logit lens. Image models use their `AutoImageProcessor` and `id2label` names.
2. **Input.** `RunManager.build_input` turns an upload (image / `.npy` / `.pt`), a text or a random spec into a
   tensor. Integer `.npy` arrays (token ids) stay `int64`; everything else becomes float32. A **text** sample is
   `tokenizer.encode(text)` as a `(1, T)` int64 tensor (400: no tokenizer, no tokens, or more than
   `min(512, n_positions)` tokens); a random sample of a model with a tokenizer is random token ids. Every `(1, T)`
   integer input of a model with a tokenizer records `sample.token_ids` and `sample.tokens` (one `decode([id])`
   each) in the meta.
3. **Capture.** The run's backend (explicit `backend`, or auto: NNsight above `NEUROTOME_NNSIGHT_PARAMS`
   parameters when nnsight is installed, else TorchLens; see Capture backends) records the forward pass.
   `capture.capture()` calls `torchlens.trace(model, x, layers_to_save="all")`, drops buffer
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
lens/<target|top>.json                              logit lens (LM runs)
```

`okey` names the objective: `top` (the top prediction), the logit index, or `u_<op>_c<channel>` for a unit.
An attribution file without `merges` (written before M3) is stale and recomputed, and so is one of an integer-input
run with `input: null` and no `tokens` key (written before v0.2).

The preview is de-normalised with the model's `(mean, std)` for uploaded images and min-max stretched per
channel for random, tensor and in-process inputs. Runs captured before M2 have no preview and no `meta.preview`;
they still open, and the UI disables the overlay.

`meta.output_error` (since M4) is the `describe()` message of the top-k failure (for example "The model returned
SimpleNamespace, which holds no tensor…"), `null` when the top-k worked; attribution fails with the same text.

Since v0.2 (all optional; readers use `.get`, so older runs open unchanged):

```jsonc
{
  "model": { /* spec, weights, the IR "model" block */
    "backend": "torchlens" | "nnsight",   // absent = torchlens
    "granularity": "op" | "module",        // module for nnsight
    "lm": true },                          // causal LM with a tokenizer: enables the logit lens; absent = false
  "sample": { "kind": "random" | "image" | "tensor" | "text", "name": "the cat sat…",
    "text": "the cat sat on the mat",      // kind text only
    "token_ids": [464, 3797],              // any (1, T) integer input of a model with a tokenizer
    "tokens": ["The", " cat"] },           // decode([id]) per id, same length as token_ids
  "categories": null                        // always null for vocabulary-sized labels (Vocab); a list otherwise
}
```

## Backend modules (`backend/neurotome/`)

| File | Responsibility |
|---|---|
| `config.py` | Settings from env: `NEUROTOME_RUNS_DIR`, `NEUROTOME_HOST`, `NEUROTOME_PORT`, `NEUROTOME_DEVICE`, `NEUROTOME_MAX_CAPTURE_MB`, `NEUROTOME_MAX_LIVE_MODELS`, `NEUROTOME_NNSIGHT_PARAMS` |
| `errors.py` | `TraceError` (422), `ModelUnavailable` (409), `NO_GRAD`, `oom_message()`, `describe(e) → (status, message)`: the one place exceptions become user-facing text (see Errors) |
| `models.py` | Model specs (`torchvision:<name>`, `hf:<repo_id>`, `py:<module>:<factory>`, random init seeded), `check_spec()`, preprocessing, tokenizers and `Vocab` (lazy token labels), `is_lm()`, `_prepare_hf()` (eager attention, no KV cache), `to_logits()` (a clear `ValueError` for outputs without float logits) |
| `capture.py` | **The only TorchLens-aware file** (backend `torchlens`, granularity `op`). Trace → IR with the never-cross budget and selective capture; per-op gradients (logit or unit objective) and merge VJPs; activation patching by DAG replay |
| `capture_nnsight.py` | **The only NNsight-aware file** (backend `nnsight`, granularity `module`, the `nnsight` extra). The same `capture` / `op_gradients` / `patch` contract at leaf-module granularity: ops are leaf-module calls (`transformer.h.0.mlp.c_fc@1`), structure, edges (tensor identity) and values from forward hooks under an NNsight trace (values copied at the hook, so a later in-place op such as `ReLU(inplace=True)` cannot change them; gradients hooked on the same pre-in-place tensors); `merges` is empty; `patch` raises 400 |
| `lens.py` | Logit lens: `lens_parts(model)` finds the blocks (`transformer.h`, `model.layers`, …), the final norm and the unembedding; `logit_lens()` reads block 0's input and every block's output through them (forward hooks, one pass, no TorchLens) |
| `store.py` | Run folders, JSON read/write (atomic), `.npy` save / memory-mapped load |
| `graph.py` | `graph_view(ir, depth, scores, expand, collapse, merges, start)` fold algorithm; `search_index(ir)`; `module_tree(ir)`; `attach_scores`; `module_site(ir, op)` / `module_calls`; `unsaved(ir, op)` (derived for old runs) |
| `tensors.py` | `tensor_view(array, channel, pixel, norm, cmap, lo, hi)` → `attention` / `image` / `sequence` / `vector` summaries; `input_preview`; `tensor_diff` |
| `slicing.py` | Regex-only NumPy-style slice grammar (no `eval`): `slice_table` (≤ 4096 values as JSON), `slice_npy` (≤ 2²⁴ values) |
| `attribution.py` | Captum input and layer methods, token attribution on the embedding layer (`embedding_layer`, `token_attribution`), objective hooks, channel conductance, method agreement, assembly of the attribution result |
| `jobs.py` | `JobRunner`: one worker thread, progress, cancellation, `wait()` for SSE; kinds `capture`, `attribution`, `layer`, `neuron`, `agreement`, `patch`, `lens`; error text through `describe` |
| `runs.py` | `RunManager`: create runs (`start_capture` job or synchronous `create_from_spec` / `create_from_objects`), text input and token meta, backend choice and dispatch (`_backend(run_id)`), keep live models, cache attribution / layer / neuron / agreement / patch / lens results, `job()` dispatcher, one-at-a-time lock |
| `api.py` | FastAPI app factory; `ValueError` / `RuntimeError` handlers through `describe()`; serves the built UI from `static/` when present |
| `cli.py` | `neurotome serve`, `neurotome capture [--text "…"] [--backend torchlens\|nnsight] [--modules a,b] [--max-capture-mb N]` |
| `__init__.py` | `neurotome.open(model, x, tokenizer=None)` for scripts and notebooks |

Optional extras: `pip install 'neurotome[hf]'` (transformers, accelerate: `hf:` specs) and
`pip install 'neurotome[nnsight]'` (the NNsight backend). Both are imported lazily; without them the rest works and
the request that needs one fails with the install command (400).

## IR (`ir.json`)

```jsonc
{
  "model": {"class_name": "ResNet", "num_params": 11689512, "num_ops": 69, "flops": 3645279184,
            "activation_bytes": 33538752,     // bytes written to tensors/
            "truncated": false,               // at least one op has unsaved == "budget"
            "budget_bytes": 2147483648,       // the budget this capture used, or null (absent before M4)
            "capture_modules": null,          // ["layer3", "layer4"] for a selective capture (absent before M4)
            "saved_ops": 69,                  // non-io ops with saved == true; compare with num_ops (absent before M4)
            "backend": "torchlens",           // or "nnsight" (absent before v0.2 = torchlens)
            "granularity": "op"},             // "module" for nnsight: ops are leaf-module calls (absent = op)
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

After dropping the batch axis (ranks above 3 fold into C; the attention test runs before the image rule):

| Kind | Shape | Payload |
|---|---|---|
| `attention` | (H, T, T), T > 1, finite, in [0, 1], every row sums to 1 (a softmax: attention probabilities) | every head (≤ 64) pooled to ≤ 64×64 and scaled to its max; the selected head (`channel`) pooled to ≤ 128×128; per-head entropy (bits) and mean weight on the previous / first / same token, with the `pattern` above 0.5; stats; histogram of the head |
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
| GET | `/api/models` | `{suggested, methods, text: ["hf:gpt2", "hf:Qwen/Qwen2.5-0.5B"], backends: {hf: bool, nnsight: bool}, nnsight_params}`: suggested specs, attribution methods, the text models, which extras are installed, the auto-NNsight threshold |
| GET | `/api/runs` | newest first |
| POST | `/api/uploads` | multipart `file` → `{upload_id}` (≤64 MB) |
| POST | `/api/runs` | `{model, weights, sample: {kind: random\|upload\|text, upload_id, text, shape, seed}, modules: [address] \| null, max_capture_mb: finite >0 \| null, backend: torchlens\|nnsight\|null}` → **202** + a `capture` Job (`run_id: null`, `result` = run meta when done); `text` is 1–10 000 chars; `backend` null = auto; 400 bad spec prefix, missing upload, `Text sample needs text`, or `nnsight` without the package; 422 bad types |
| GET / DELETE | `/api/runs/{id}` | meta / delete run folder |
| GET | `/api/runs/{id}/modules` | module tree for the left rail |
| GET | `/api/runs/{id}/graph?depth=2[&expand=a:1,b:1&collapse=c:1][&method=ig&target=207&steps=32]` | graph view; `expand` / `collapse` are comma-separated module calls; with `method`, scores, shares and path attached |
| GET | `/api/runs/{id}/index` | search index: every op, then every multi-op module call, as `{id, kind, label, path, type, shape, calls}` |
| GET | `/api/runs/{id}/ops/{op}` | one IR op |
| GET | `/api/runs/{id}/ops/{op}/tensor?channel=&py=&px=&norm=channel\|global&cmap=seq\|div&lo=&hi=` | tensor view; for `kind: attention` `channel` selects the head; bad `norm` / `cmap` → 422; no tensor → 404 (below) |
| GET | `/api/runs/{id}/ops/{op}/slice?s=<expr>` | slice as a JSON table (≤ 4096 values); bad expression → 400 with a readable `detail` |
| GET | `/api/runs/{id}/ops/{op}/slice.npy?s=<expr>` | the whole slice as a float32 `.npy` download; 400 bad expression, 413 over 2²⁴ values |
| GET | `/api/runs/{id}/diff?a=<op>&b=<op>` | `{a, b, shape_a, shape_b, same_shape, diff}`; 404 when either tensor is missing |
| GET | `/api/runs/{id}/input.png` | input preview (PNG); 404 for runs without one |
| GET | `/api/runs/{id}/graph?…&obj_op=<op>&obj_ch=<c>` | with a unit objective (both or neither, else 400): scores, merge shares and a path that starts at the unit |
| POST | `/api/runs/{id}/attribution` | `{method: ig\|saliency\|gxa, target, steps, objective?: {op, channel}}` → cached result (synchronous); 400 on a bad objective. Integer inputs get `tokens` (below) and `input: null` |
| POST | `/api/runs/{id}/jobs` | `{kind: attribution\|layer\|neuron\|agreement\|patch\|lens, method, target, objective, steps, op, channel, baseline: zeros\|mean\|run, source_run}` → Job; 404 unknown run, 400 missing fields, 422 bad kind / types. `lens`: `target` is a token id or null (the top prediction), result `LensResult`; 400 for a run that is not an LM. `patch` on an NNsight run: 400 (`Activation patching replays TorchLens ops; …`) |
| GET | `/api/jobs/{job_id}` | `{id, kind, run_id, status: queued\|running\|done\|error\|cancelled, progress, message, result, error, created}`; 404 unknown |
| POST | `/api/jobs/{job_id}/cancel` | the job after the cancel request |
| GET | `/api/jobs/{job_id}/events` | `text/event-stream`: `data: <job json>` on every change, `: ping` every 15 s, ends after a terminal state |

**Token attribution** (`tokens`, for a token-id input with an embedding layer): the chosen method applied to the
embedding layer's output, one score per position: Captum `LayerIntegratedGradients` from a baseline of **token
ids** (the pad token, else end-of-sequence, else 0), `LayerGradientXActivation`, or its gradient alone for saliency.

```jsonc
{"scores": [0.12, -0.03],          // Σ over the hidden dimension; saliency: L2 norm of the gradient (unsigned)
 "signed": true, "total": 0.09, "convergence_delta": 3e-8,   // Δ for IG only
 "layer": "transformer.wte", "baseline_id": 50256,           // baseline_id: IG only
 "method_label": "Layer IG on transformer.wte · baseline token 50256"}
```

**Logit lens** (`LensResult`, `lens.py`): one forward pass with a pre-hook on block 0 (row `embed`) and a hook on
every block; each row is `softmax(head(norm(resid)))` for the last ≤ 64 positions, so the last block's row equals the
model's own output. HF applies exactly that final norm and head, which the tests check for GPT-2 and Llama.

```jsonc
{"method_label": "Logit lens · transformer.ln_f + lm_head",
 "layers": ["embed", "transformer.h.0", "transformer.h.1"], "pos0": 0, "tokens": ["The", " cat"], "k": 5,
 "top": [[[{"id": 262, "token": " the", "prob": 0.41}]]],   // [layer][position][rank]
 "target": {"id": 3290, "token": " mat"},                   // default: the model's top prediction at the last position
 "target_prob": [[0.003]], "target_rank": [[17]]}           // [layer][position]; rank 1 = top
```

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
| `ValueError` | 400 | its text: bad requests, `Cannot load model '<spec>': …`, `Unknown module(s) …`, `The model returned <type>, which holds no tensor…`, integer inputs for layer methods (`…token inputs get Layer IG on the embeddings in the saliency panel instead.`), and the v0.2 messages below |
| `RuntimeError` naming a missing gradient (`does not require grad`, `…not have been used in the graph`) | 400 | `NO_GRAD`: forward() leaves autograd between the input and the output, or a frozen / parameter-free model has an integer input (`op_gradients` makes float inputs require grad, so frozen models with float inputs still get node gradients) |
| anything else | 500 | `<Type>: <message>` |

An empty message is replaced by the exception's type name. A failed capture job deletes its half-written run folder;
after an OOM it also empties the CUDA cache. Integer inputs (token ids) get node scores and, when the model has an
embedding layer, `tokens` (token attribution) with `input: null` and `input_note: null`; without one the result
carries `input: null`, `tokens: null` and an `input_note` that says why.

v0.2 messages (all 400):

| Where | Message |
|---|---|
| `hf:` spec without transformers | `Hugging Face models need the hf extra: pip install 'neurotome[hf]'` |
| `backend: nnsight` without nnsight | `The nnsight backend needs the nnsight extra: pip install 'neurotome[nnsight]'` |
| text sample | `Text sample needs text` (at POST); `This model has no tokenizer; upload a .npy of token ids or a tensor instead.`; `The text has no tokens`; `The text is {n} tokens; this model takes at most {limit}` |
| lens | `Logit lens needs a causal language model with a tokenizer; this run's model is not one.`; `Logit lens needs token ids as input`; `Logit lens needs a causal LM whose final norm and unembedding it can find (GPT-2, Llama, Qwen, Mistral, GPT-NeoX, Phi); <Class> is not supported.` |
| patch on an NNsight run | `Activation patching replays TorchLens ops; this run was captured with NNsight (module outputs only). Capture it again with the TorchLens backend to patch.` |

## Jobs (`jobs.py`)

Captures and attribution, layer, neuron, agreement, patch and lens requests run as jobs on **one** daemon worker thread fed by a
queue, so one job runs at a time. Jobs call `RunManager` methods, which take `RunManager._lock` themselves (the
worker never does; the lock is not re-entrant), so a synchronous capture and a job never share the GPU.
`RunManager.job()` validates the request, answers a cache hit with a job born `done`, and otherwise submits the
work keyed by its cache file, so a second identical request joins the queued or running job (never one already asked to cancel). Work reports
`progress(fraction, message)` (`Node gradients`, `IG batch i/n`, `Layer IG batch i/n`, `Conductance batch i/n`,
`Agreement · <method> (i/N)`, `Clean pass`, `Patched replay`, `Lens forward`, `Project layer i/L`); the fraction is monotone and held ≤ 0.99 until
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

**Lens** (`RunManager.lens`, kind `lens`) runs `lens.logit_lens` on the live model and the run's `(1, T)` token
ids and caches `lens/<target|top>.json`.

## Capture backends: TorchLens vs NNsight

A run records its backend in `meta.model.backend` / `ir.model.backend`. `RunManager._backend(run_id)` imports that
module lazily, and attribution uses its `op_gradients` and patching its `patch`, so op ids always match the IR.
Choice at capture: an explicit `backend`, else **auto**: `nnsight` when the model has more parameters than
`NEUROTOME_NNSIGHT_PARAMS` (default 1e9) and nnsight is installed (the capture job then reports
`Large model (n B params) · NNsight module capture`), else `torchlens`.

| | TorchLens (default) | NNsight (`backend: nnsight`, auto above 1 B params) |
|---|---|---|
| Graph | every op, including functional ones (`add`, `softmax`, `view`) | leaf-module calls; functional ops between modules are invisible and edges fall back to call order |
| Attention maps | every softmax (not SDPA: torchvision `vit_b_16` fuses attention, so it has none) | only where a module outputs the probabilities (GPT-2 `attn_dropout`, which transformers 5 calls as a module only with `reorder_and_upcast_attn=True`); stock GPT-2, Llama and Qwen apply softmax and dropout functionally, so none |
| Merge shares | exact grad × input through merges | parent shares (approximate, labelled `share_method: parent`) |
| Patching | yes (DAG replay) | no (400) |
| Cost | records every op (memory and time grow with op count) | hooks + one float32 copy per kept leaf call; much smaller IR |

The logit lens and token attribution use plain hooks and Captum on the live model, so they work with either backend.

Interactive docs live at `http://127.0.0.1:7070/docs` while the server runs.

## Configuration and CLI

`NEUROTOME_RUNS_DIR` (default `./runs`), `NEUROTOME_HOST` / `NEUROTOME_PORT` (`127.0.0.1:7070`),
`NEUROTOME_DEVICE` (`cpu`), `NEUROTOME_MAX_CAPTURE_MB` (the per-run activation budget, default 2048),
`NEUROTOME_MAX_LIVE_MODELS` (default 2) and `NEUROTOME_NNSIGHT_PARAMS` (auto backend: NNsight above this many
parameters, default 1e9).

- `neurotome serve [--host] [--port] [--runs-dir] [--device] [--reload]`
- `neurotome capture MODEL [--weights] [--image | --text "…"] [--backend torchlens|nnsight] [--runs-dir]
  [--modules layer3,layer4] [--max-capture-mb 512]`: synchronous; `--text` is a text sample (models with a
  tokenizer), `--backend` overrides the auto choice, `--modules` is comma-separated (whitespace stripped),
  `--max-capture-mb` overrides the budget for this run.
- `neurotome.open(model, x, tokenizer=None)` captures in-process (synchronously) and serves the UI; a tokenizer
  labels token ids and, for a causal LM, enables the logit lens.
- Extras: `pip install 'neurotome[hf,nnsight]'`.

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `lib/api.ts`, `lib/types.ts` | Typed client for the API above |
| `lib/router.ts` | Hash routes `#/runs/<id>/<graph\|tensors\|trace\|lens>?node=…` (shareable deep links; the graph view also keeps `depth`, `expand`, `collapse`, `color` and the patch `patch`/`pb`/`psrc`) |
| `lib/tokens.ts` | Token labels: `showToken` (visible whitespace), `tokenLabels(run, n)`, `spanLabel` for pooled cells, `tokenTone` for the token strip |
| `lib/lensState.ts` | Lens URL state (`t`, `mode`, `pos`), cell text, readout, `shortLayer` |
| `lib/attentionState.ts` | Attention view helpers: `headReadout`, `patternLabel`, `axisLetters` (`B H Q K`) |
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
| `components/ComparePanel.tsx` | Compare two ops on a shared scale, with the diff statistics (attention ops show the selected head side by side) |
| `components/panels.tsx` | Module tree, node inspector (with `Trace from` channel link, `Layer attribution` link and the patch panel) |
| `components/PatchPanel.tsx` | Patch baseline choice, `Patch & re-run`, Δlogit / Δprob and top-5 after the patch (hidden for NNsight runs) |
| `components/JobProgress.tsx` | Job progress line with `Cancel`; the error text of a failed job |
| `components/CaptureBanner.tsx` | The `Truncated` / `Selective` row under the TopBar for partial captures (Graph and Tensor views) |
| `components/AgreementMatrix.tsx` | n×n method-agreement matrix (Jaccard or Spearman) with a hover readout and PNG export |
| `components/TokenStrip.tsx` | Token attribution as inline tokens tinted by score (group `Token attribution`), with a readout |
| `views/LensView.tsx` | Logit lens grid (layers × positions, `Top-1` / `Target` colour), target across layers, top-k at a cell |
| `views/*` | Home (text / file sample, capture backend), Graph, Tensors (incl. the attention view), Trace (incl. the token strip), Lens |
| `styles/tokens.css` | Design tokens (mirrors `design/tokens.json`) |

Tensor view URL: `#/runs/<id>/tensors?node=…&c=…&py=…&px=…&norm=global&cmap=div&cmp=…` (defaults omitted; `cmp` turns on
compare mode). Overlay on/off and opacity are view state, not URL state.

Trace view URL: `#/runs/<id>/trace?node=…&m=ig&t=207&obj=<op>&ch=<c>&steps=32&layer=layer_ig&lop=<op>&nc=<c>&go=1`
(defaults omitted). `go=1` runs the attribution, then the layer and neuron jobs, once and drops itself; the
Tensor view and the graph inspector link here with it.

Lens URL: `#/runs/<id>/lens?t=<token id>&mode=target&pos=<position>` (defaults omitted: `t` = the top prediction,
`mode` = `top`, `pos` = the last position). The Lens tab is disabled for runs whose `model.lm` is not true.

Dev: Vite on :5173 proxies `/api` to :7070. Build: output goes to `backend/neurotome/static/`, so the Python package serves the UI;
`index.html` is served with `Cache-Control: no-cache` so a rebuilt UI never points at deleted hashed assets. Every other path
falls back to `index.html`, except `/api/*` (a JSON 404) and paths that resolve outside `static/` (never served).

## Extension points

- **New capture backend** (`torch.export`, forward hooks; `capture_nnsight.py` is the worked example): implement
  `capture()`, `op_gradients()` (including the `merges` field and the unit objective) and `patch()`, returning the
  same IR, score and delta dicts, and name it in `runs._module` / `BACKENDS`. Nothing downstream changes.
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
