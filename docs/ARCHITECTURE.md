# Architecture

## Flow of one run

0. **Capture job.** `POST /api/runs` validates what needs no model (the spec prefix, the upload's existence; 400
   otherwise) and queues a `capture` job (`RunManager.start_capture`); it answers `202` with the job at once. The
   job reports `Load model`, `Build input`, `Trace`, `Save tensors i/N`, `Summarise` and ends with the run meta as
   its `result`; the browser follows it like any other job (SSE or polling, `Cancel`). A failed or cancelled
   capture leaves no run folder. The CLI and `neurotome.open()` call the same code synchronously.
1. **Load.** `models.load_model(spec, weights)` returns the `nn.Module` plus preprocessing and class names
   (`torchvision:<name>`, `hf:<repo_id>` or `py:<module>:<factory>`). Random-init weights are built under a fixed
   seed, so reloading a spec later (attribution after a restart or an LRU eviction, interventions, source runs)
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
   - **`py:` module hooks** (read with `getattr`, all optional): `INPUT_SHAPE` (the random sample's shape),
     `CATEGORIES` (class names), `TOKENIZER`, `preprocess(img)` (image → batched tensor), `OUTPUT_KIND`
     (`"logits"` by default; `"values"` for score / regression outputs: the output summary is the top-5 by raw
     value with `prob: null`, no softmax), and a dataset: `list_samples() -> [{ref, label, group}]` (called by
     `models.list_samples(spec)`, which only imports the module and caches the list per spec for the server's
     life) and `load_sample(ref) -> dict` (below), and `target_windows(index) -> [{label, start, stop, blocks,
     kind}]`: the regions of the input's time axis that output element `index` reads (`kind` `target` = the step
     itself, `context` = what it may read, `horizon` = the future it predicts; `blocks` names input blocks, null =
     every channel). `models.target_windows(spec, index)` only imports the module; `runs.check_windows` checks the
     return against the run's input (`GET /windows`, below). `examples/signal_model.py` is the worked example.
2. **Input.** `RunManager.build_input` turns an upload (image / `.npy` / `.pt`), a text or a random spec into a
   tensor. Integer `.npy` arrays (token ids) stay `int64`; everything else becomes float32. A **text** sample is
   `tokenizer.encode(text)` as a `(1, T)` int64 tensor (400: no tokenizer, no tokens, or more than
   `min(512, n_positions)` tokens); a random sample of a model with a tokenizer is random token ids. Every `(1, T)`
   integer input of a model with a tokenizer records `sample.token_ids` and `sample.tokens` (one `decode([id])`
   each) in the meta.
   A **dataset sample** (`{kind: "ref", ref}`, ref ≤ 512 chars) is the model module's `load_sample(ref)`, a dict:
   `input` (a tensor or tuple of tensors, batch 1; required), `name`, `steps` (the transform raw → input in words),
   `info` (metadata rows: str / int / float / null; anything else is stringified), `raw` (`{fs, t0, time_label,
   channels: [{name, unit, values}]}`, the signal before the transform), `blocks` (`[{name, start, stop, labels,
   unit}]`, column groups of a (1, T, C) input's last axis) and `time` (`{axis, step_s, t0}`: step i of that axis
   sits at `t0 + i · step_s` on the raw signal's clock). `runs._check_sample` checks every key and raises a
   `ValueError` naming what is wrong (400 / the capture job's error); an exception inside `load_sample` becomes
   `load_sample('<ref>') failed: <type>: <message>`. A run of a dataset sample reloads like any `py:` run (the
   model by spec, the input from `inputs.pt`); `load_sample` is not called again.
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
   one channel of a module's output; the **channel axis** is the features of a `(B, T, D)` tensor and axis 1
   otherwise, as for interventions, so per-channel scores, layer methods and neuron conductance of a token tensor
   are per feature). `attribution.attribute()` re-traces with gradients, back-propagates the
   objective, and returns grad×activation for every op plus each parent's grad×input through every merge
   (`merges`). It also runs a Captum method on the input, with a forward hook on the unit's module for a unit
   objective. Layer IG / conductance, neuron conductance and method agreement work the same way. The Trace view
   starts them as **jobs** (`POST /api/runs/{id}/jobs`) and follows their progress over SSE; results are cached in
   `attribution/` (below), so a repeated request is answered at once.
6. **Intervene** (on request, as a job). An intervention set (below) edits op outputs (zero, mean, resample,
   constant, scale, steer, scoped by channels and positions) in **one forward pass**: the run's capture backend
   re-runs the model with the edits applied where each op runs (`intervene`), and every downstream op gets a
   before/after diff row and its intervened stats; the downstream tensors are saved under the run's capture budget
   in `interventions/<hash>/`. TorchLens runs use **capture-time hooks** (a clean trace for each op's `raw_index`,
   then one edited `tl.trace(..., intervene=pick)`), not DAG replay: replay stops at multi-pass ops such as GPT-2's
   residual adds (two passes of one layer), so downstream values after them were never recomputed. An edit of an
   op that returns its input object (eval-mode dropout, which TorchLens records unchanged) moves to that input when
   the op is the input's only consumer, which is the same edit (GPT-2's `attn` / `mlp` module nodes end in one);
   otherwise it is refused. LM runs can also
   **generate**: greedy continuations, clean and with the edits as forward hooks on module outputs.

## Run folder (`runs/<run_id>/`)

```
meta.json            run summary: model spec + the IR "model" block, sample, output top-k, output_error,
                     preview {width, height} | null
output.npy           the whole output vector (to_logits of sample 0, float32); runs before it get one on the
                     first GET /output (a forward pass of the live model)
ir.json              the captured graph (below)
inputs.pt            the exact input tensors, so attribution can re-run later
input_preview.png    the input as a picture, <= 224 px; only for (N, 1|3, H, W) inputs with H, W >= 8
input_raw.<ext>      the uploaded image as it was (jpg / png / gif / webp / bmp; other formats as png), 0.3.0+
input_raw.npz        a dataset sample's raw signal: one float32 array per channel, c0, c1, …
tensors/<op>.npy     one activation per saved op, float32, memory-mapped on read
attribution/<method>_<okey>_<steps>.json            input + node attribution (with merges, objective)
attribution/channels_<okey>.json                    per-op per-channel g×a (server-side only)
attribution/<layer_ig|layer_cond>_<okey>_<op>_<steps>.json
attribution/neuron_<okey>_<op>_c<ch>_<steps>.json
attribution/agreement_<okey>_<steps>[_<op>].json
lens/<target|top>.json                              logit lens (LM runs)
interventions/<hash>/meta.json                      InterventionResult (below); rename rewrites "name" only
interventions/<hash>/tensors/<op>.npy               intervened activations of saved downstream ops (float32)
interventions/<hash>/generate_<n>.json              GenerationResult for n new tokens
```

`okey` names the objective: `top` (the top prediction), the logit index, or `u_<op>_c<channel>` for a unit.
An attribution file without `merges` (written before M3) is stale and recomputed, and so is one of an integer-input
run with `input: null` and no `tokens` key (written before v0.2), and an attribution / neuron / agreement file
whose `input_map` is not `"series-512"` (a sequence input's map was a 24×24 grid, then time pooled to 128
columns) is recomputed too. Old `patch/*.json` files (M3) are ignored. Runs
before v0.3 have no `interventions/` folder: their list is empty. Deleting a run deletes its interventions.

The preview is de-normalised with the model's `(mean, std)` for uploaded images and min-max stretched per
channel for random, tensor and in-process inputs. Runs captured before M2 have no preview and no `meta.preview`;
they still open, and the UI disables the overlay.

`meta.input` (since 0.3.0; absent before) is `{raw: {file, width, height, format, bytes} | null, steps: [str] | null,
norm: [mean, std] | null}`: the uploaded image kept as `input_raw.<ext>`, the preprocessing in words, and the
per-channel `(mean, std)` it normalised with (so `/input/model.png` can undo it). A dataset sample adds
`blocks` and `time` as `load_sample` gave them (checked, JSON-safe; `null` otherwise), and its raw is
`{kind: "signal", file: "input_raw.npz", fs, t0, time_label, channels: [{name, unit, length}]}` (an image raw has
no `kind`). Its `meta.sample` is `{kind: "ref", ref, name, info}`.

`meta.output_error` (since M4) is the `describe()` message of the top-k failure (for example "The model returned
SimpleNamespace, which holds no tensor…"), `null` when the top-k worked; attribution fails with the same text.

Since v0.2 (all optional; readers use `.get`, so older runs open unchanged):

```jsonc
{
  "model": { /* spec, weights, the IR "model" block */
    "backend": "torchlens" | "nnsight",   // absent = torchlens
    "granularity": "op" | "module",        // module for nnsight
    "lm": true },                          // causal LM with a tokenizer: enables the logit lens; absent = false
  "sample": { "kind": "random" | "image" | "tensor" | "text" | "ref", "name": "the cat sat…",
    "ref": "rec01:0", "info": { "recording": "rec01" },   // kind ref only (a py: module's load_sample)
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
| `capture.py` | **The only TorchLens-aware file** (backend `torchlens`, granularity `op`). Trace → IR with the never-cross budget and selective capture; per-op gradients (logit or unit objective) and merge VJPs; `intervene` (capture-time hooks matched on `raw_index`, one clean and one edited trace) |
| `capture_nnsight.py` | **The only NNsight-aware file** (backend `nnsight`, granularity `module`, the `nnsight` extra). The same `capture` / `op_gradients` / `intervene` contract at leaf-module granularity: ops are leaf-module calls (`transformer.h.0.mlp.c_fc@1`), structure, edges (tensor identity) and values from forward hooks under an NNsight trace (values copied at the hook, so a later in-place op such as `ReLU(inplace=True)` cannot change them; gradients hooked on the same pre-in-place tensors); `merges` is empty; `intervene` edits leaf calls (`env.output = fn(env.output)`, `tracer.iter[k − 1]` for call k) |
| `interventions.py` | Plain torch / numpy (no TorchLens, no NNsight): `normalise` (canonical set and the 400 messages), `set_hash`, `downstream`, `keep_ops` (never-cross budget), `scope_mask`, `edit_fn` (what an edit computes), `label`, `diff_row`, `generate` (greedy decoding with forward hooks at module sites) |
| `lens.py` | Logit lens: `lens_parts(model)` finds the blocks (`transformer.h`, `model.layers`, …), the final norm and the unembedding; `logit_lens()` reads block 0's input and every block's output through them (forward hooks, one pass, no TorchLens) |
| `store.py` | Run folders, JSON read/write (atomic), `.npy` save / memory-mapped load (`iv=` reads and writes `interventions/<hash>/tensors/`; `iv_dir` raises `KeyError` unless the hash is 12 hex digits) |
| `graph.py` | `graph_view(ir, depth, scores, expand, collapse, merges, start)` fold algorithm; `search_index(ir)`; `module_tree(ir)`; `attach_scores`; `module_site(ir, op)` / `module_calls`; `unsaved(ir, op)` (derived for old runs) |
| `tensors.py` | `tensor_view(array, channel, pixel, norm, cmap, lo, hi)` → `attention` / `image` / `sequence` / `vector` summaries; `input_preview`; `tensor_diff` |
| `slicing.py` | Regex-only NumPy-style slice grammar (no `eval`): `slice_table` (≤ 4096 values as JSON), `slice_npy` (≤ 2²⁴ values) |
| `attribution.py` | Captum input and layer methods, token attribution on the embedding layer (`embedding_layer`, `token_attribution`), objective hooks, channel conductance, method agreement, assembly of the attribution result |
| `jobs.py` | `JobRunner`: one worker thread, progress, cancellation, `wait()` for SSE; kinds `capture`, `attribution`, `layer`, `neuron`, `agreement`, `lens`, `intervene`, `generate`; error text through `describe` |
| `runs.py` | `RunManager`: create runs (`start_capture` job or synchronous `create_from_spec` / `create_from_objects`), text input and token meta, backend choice and dispatch (`_backend(run_id)`), keep live models, cache attribution / layer / neuron / agreement / lens results, interventions (`resolve` source runs, `intervene`, `generate`, list / rename / delete, `iv_tensor`, `iv_ir`), `job()` dispatcher, one-at-a-time lock |
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
   score| / Σ, labelled `share_method: parent`. A single incoming edge has `share` 1 and no method. A node no flow
   reaches (score 0 or missing, e.g. after a unit objective) has no `share_method`: its edges keep the parent
   shares (1/n when no parent has a score) only so they can be drawn. The
   **dominant path** is a greedy walk along the highest-share parent from the output, or from the unit's node
   for a unit objective (`start`).

> For linear merges (add, sub, cat, stack) the contributions of an op sum exactly to its own grad×activation, so
> the shares split the merge's flow. A residual's identity branch is credited only with what flows through the
> add, not with its own total (which also counts flow through the conv branch).

## Tensor views (`tensors.py`)

After dropping the batch axis (ranks above 3 fold into C; the attention test runs before the image rule):

| Kind | Shape | Payload |
|---|---|---|
| `attention` | 3-D, K > 1, finite, in [0, 1], every row along the last axis sums to 1 ± 1e-2 (softmax, or entmax / sparsemax with exact zeros) or is all zero (a fully masked row), with at least half the rows summing to 1. `layout`: (H, T, T) is `hqk`; otherwise the heads are the smaller of the first two axes, ≤ 64 (ties heads-first): (H, Q, K) `hqk`, or (Q, H, K) `qhk` (heads in the middle, e.g. lag attention `(time, head, lag)`). A spatial softmax over W of a (C, H, W) map with C ≤ H therefore reads as C heads | `layout`, `square`, `heads`, `queries`, `keys` (`tokens` = Q); every head (≤ 64) pooled to ≤ 64×64 and scaled to its max; the selected head (`channel`) pooled to ≤ 128×128; per head over its live (not all-zero) rows: entropy (bits) and `entropy_norm` (÷ log₂ K), `sparsity` (fraction of exact zeros), `masked` (fraction of all-zero rows), `argmax` / `argmax_weight` (peak of the key profile); `key_profile` per head (mean weight per key over live queries, pooled to ≤ 256; for lag attention the lag profile). Square maps also get the mean weight on the previous / first / same token over queries 1…T−1 (query 0's only key under a causal mask is itself and the first token), with the `pattern` above 0.5; stats; histogram of the head |
| `image` | (C, H, W), H, W > 1 (higher ranks fold into C) | top-32 channels by mean as 24×24 pooled maps; selected channel as 16×16 dots; peak-row profile; channel vector at a pixel; PCA of the H·W positions (≤400 points); histogram |
| `sequence` | (T, D) | heatmap pooled to ≤64×128 (signed); per-token L2 norm; PCA of tokens; histogram |
| `vector` | (D,) | values pooled to ≤512; top-10 by magnitude; histogram |

Everything is pooled on the server, so no response carries more than a few thousand numbers. View `stats`,
histograms, `peak` and the vector `top` list read the finite values only (`null` when there are none, and for a
non-finite `top` value); the plotted grids replace NaN with 0 and clamp ±inf to the finite range, so one NaN never
blanks or rescales a chart (the values table shows the raw `nan` / `inf`).

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
| GET | `/api/samples?model=<spec>` | `{samples: [{ref, label, group}], error: str \| null}`: a `py:` module's `list_samples()` (import only, cached per spec); `[]` for other specs and modules without the hook; an import or hook failure answers 200 with `samples: []` and its message |
| POST | `/api/uploads` | multipart `file` → `{upload_id}` (≤64 MB, read in bounded chunks); the id is a content hash with the suffix `.pt`, `.pth`, `.npy` or `.img` (anything else; images are recognised by content), so an upload is never importable as a `py:` module |
| POST | `/api/runs` | `{model, weights, sample: {kind: random\|upload\|text\|ref, upload_id, text, ref, shape, seed}, modules: [address] \| null, max_capture_mb: finite >0 \| null, backend: torchlens\|nnsight\|null}` → **202** + a `capture` Job (`run_id: null`, `result` = run meta when done); `text` is 1–10 000 chars; `ref` ≤ 512 chars (a dataset sample: `load_sample(ref)`, checks above); `backend` null = auto; 400 bad spec prefix, missing upload, `Text sample needs text`, `A dataset sample needs its ref`, or `nnsight` without the package; 422 bad types |
| GET / DELETE | `/api/runs/{id}` | meta / delete run folder |
| GET | `/api/runs/{id}/modules` | module tree for the left rail |
| GET | `/api/runs/{id}/graph?depth=2[&expand=a:1,b:1&collapse=c:1][&method=ig&target=207&steps=32][&iv=<hash>]` | graph view; `expand` / `collapse` are comma-separated module calls; with `method`, scores, shares and path attached (`steps` 1–512, else 422); with `iv`, ops the intervention recorded carry its stats (colour modes, the inspector histogram) |
| GET | `/api/runs/{id}/index` | search index: every op, then every multi-op module call, as `{id, kind, label, path, type, shape, calls}` |
| GET | `/api/runs/{id}/ops/{op}` | one IR op |
| GET | `/api/runs/{id}/ops/{op}/tensor?channel=&py=&px=&norm=channel\|global&cmap=seq\|div&lo=&hi=&iv=` | tensor view; for `kind: attention` `channel` selects the head; bad `norm` / `cmap` → 422; no tensor → 404 (below); `iv` below |
| GET | `/api/runs/{id}/ops/{op}/slice?s=<expr>&iv=` | slice as a JSON table (≤ 4096 values); bad expression → 400 with a readable `detail` |
| GET | `/api/runs/{id}/ops/{op}/slice.npy?s=<expr>&iv=` | the whole slice as a float32 `.npy` download; 400 bad expression, 413 over 2²⁴ values |
| GET | `/api/runs/{id}/diff?a=<op>&b=<op>&a_iv=&b_iv=` | `{a, b, shape_a, shape_b, same_shape, diff}`; each side read through its intervention when given (clean vs intervened for one node: `a=b=op&b_iv=<hash>`); 404 when either tensor is missing |
| GET | `/api/runs/{id}/input.png` | input preview (PNG); 404 for runs without one |
| GET | `/api/runs/{id}/input` | the Input view: `{op, shape, dtype, sample, raw: {file, width, height, format, bytes} \| null, steps: [str] \| null, normalised, picture, channels: [{channel, mean, std, min, max}]}`; `steps` is the preprocessing in words (from the torchvision transform, the default pipeline, the Hugging Face image processor, the tokenizer, or `Custom: <module>.preprocess`); `channels` are over the finite values of sample 0, per channel for an (N, C, H, W) input (≤ 64), one row per block (with its `name`) for a (1, T, C) input with `blocks`, else one row with `channel: null`; old runs give `raw` / `steps` null. Also `blocks`, `time` (from `meta.input`) and `info` (`meta.sample.info`), null when absent |
| GET | `/api/runs/{id}/input/raw` | the raw uploaded image (`input_raw.<ext>`), or a dataset signal's `input_raw.npz` (`application/octet-stream`); 404 for runs without one (random, text, tensor inputs, runs before 0.3.0) |
| GET | `/api/runs/{id}/input/signal?points=1500` | a dataset sample's raw signal pooled with numpy to ≤ `points` (16–20 000) per channel: `{fs, t0, time_label, stride, channels: [{name, unit, length, stride, min: [], max: [], mean: []}]}`; point k covers raw samples `[k·stride, (k+1)·stride)`, NaN ignored (an all-NaN point is null), 5 significant digits; 404 without a signal raw |
| GET | `/api/runs/{id}/input/matrix` | a float (1, T, C) input as heatmaps: `{time, steps: T, stride, blocks: [{name, unit, labels, start, lo, hi, values}]}`; `values[row][col]` = block column `start + row`, time mean-pooled to ≤ 600 columns of `stride` steps; `lo` / `hi` the block's unpooled range; no `blocks` → one block `input` over every column; 404 for any other input |
| GET | `/api/runs/{id}/input/model.png` | the model input at its own resolution (≤ 1024 px), de-normalised with the run's `(mean, std)` or min–max stretched; 404 unless it is a picture |
| GET | `/api/runs/{id}/graph?…&obj_op=<op>&obj_ch=<c>` | with a unit objective (both or neither, else 400): scores, merge shares and a path that starts at the unit |
| GET | `/api/runs/{id}/output` | every output element for the step picker: `{n, stride, values, labels}`; `values` are `to_logits` of sample 0 (5 significant digits, non-finite → null), `labels` the run's `categories` (null without); above 4096 elements `values` is the max of each block of `stride` and `labels` is null |
| GET | `/api/runs/{id}/windows?t=<int ≥ 0>` | `{windows: [{label, start, stop, blocks, kind}], error: str \| null}`: the run's `py:` module's `target_windows(t)` (import only; `[]` without the hook, for other specs and in-process runs). `runs.check_windows`: a list of dicts; `start` / `stop` ints, clipped to [0, T) of the input's time axis (`meta.input.time.axis`; else the series map's `time_axis` for a 3-D input; else the longer non-batch axis), windows left empty dropped; `kind` one of target / context / horizon; `blocks` null or names of the input's blocks as the series map groups them (`attribution.series_blocks`: `meta.input.blocks` when time is axis 1, else one block per raw signal channel named like the channel when their count is C, else one block `input`); `label` defaults to `<kind> <a>–<b>`. A hook that raises (`target_windows(3) failed: KeyError: 3`) or returns anything else (`window 0 names unknown block 'fhr_st'; the input's blocks are …`) answers 200 with `windows: []` and the message; 422 for t < 0, 404 unknown run |
| POST | `/api/runs/{id}/attribution` | `{method: ig\|saliency\|gxa, target, steps, objective?: {op, channel}}` → cached result (synchronous); 400 on a bad objective. Integer inputs get `tokens` (below) and `input: null` |
| POST | `/api/runs/{id}/jobs` | `{kind: attribution\|layer\|neuron\|agreement\|lens\|intervene\|generate, method, target, objective, steps, op, channel, interventions, max_new_tokens}` → Job; 404 unknown run, 400 missing fields, 422 bad kind / types / unknown fields. `lens`: `target` is a token id or null (the top prediction), result `LensResult`; 400 for a run that is not an LM. `intervene`: `interventions` (an InterventionSet) → `Job<InterventionResult>`; `generate`: `interventions`, `max_new_tokens` 1–64 (default 16) → `Job<GenerationResult>`; both answer 400 for everything that needs no model (normalisation, source runs, saved tensors, shapes; generation kinds and sites), and a cached set is a job born `done` (the job key is the cache file). The M3 kind `patch` and its `baseline` / `source_run` fields are gone (422) |
| GET | `/api/runs/{id}/interventions` | `InterventionSummary[]` (`hash, name, created, labels, target, target_label, delta_logit, delta_prob, truncated`), newest first |
| GET / PATCH / DELETE | `/api/runs/{id}/interventions/{hash}` | the InterventionResult; PATCH `{name}` (1–80 chars; blank → 400 `Name cannot be empty`) renames it; DELETE → `{deleted: true}`; 404 `Intervention {hash} not found in {run}` |
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

**Input map** (`input`, a float input; neuron conductance's `input` too): signed and divided by its max |cell|, so
in [−1, 1]. An image (or any input of another rank) is summed over channels and mean-pooled to ≤ 24×24:
`{map, total, convergence_delta}`. A **sequence** — any 3-D float input, `(1, T, C)` or `(1, C, T)` — is a
channel × time map (`input_map` → `series_map`). The time axis is `meta.input.time.axis`, else axis 1 when
`load_sample` gave `blocks`, else the longer axis. Rows are channels in index order, cells are **sums** over
`stride` = ⌈T / 512⌉ steps and `row_stride` = ⌈C / 160⌉ channels, so up to 512 steps and 160 channels every cell is
one input element (a (1, 300, 126) input is a 126 × 300 map). Cells carry 3 significant digits (not decimals, so the
many small cells of an unpooled map still add up) and `scale` is the divisor (`map × scale` = Σ a of a cell). Blocks are `meta.input.blocks` (time on axis 1), else one per raw signal channel when their count is C (the
raw `(1, 2, 5280)` input of a model that transforms in the graph), else one block `input`:

```jsonc
{"kind": "series", "map": [[0.12, -1.0, …]],        // [ceil(C / row_stride)][ceil(T / stride)], 3 sig. digits
 "total": 0.41, "convergence_delta": 0.002,          // Σ a over the whole input; Δ for IG only
 "channels": 126, "row_stride": 1, "steps": 300, "stride": 1, "scale": 0.031,
 "time": {"t0": -1198.0, "step_s": 4.0, "label": "s to delivery"},   // step i at t0 + i · step_s; null: steps only
 "blocks": [{"name": "fhr_st", "start": 0, "stop": 36, "labels": ["ψ3 · 0.887 Hz", …], "unit": null,
             "sum": 0.2, "abs": 1.3, "share": 0.44}],               // Σ a, Σ|a|, Σ|a| / Σ|a| of all channels
 "time_totals": [0.01, -0.03, …]}                    // signed Σ over channels of each column
```

`time` is `meta.input.time` when its axis is the time axis, else the raw signal's clock (`t0`, `1 / fs`, its
`time_label`) when every raw channel is T samples long (the input is the raw signal), else `null`. Shares sum to 1
when the blocks cover every channel. Agreement over such maps names its basis `channel × time input map`.

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

With `iv=<hash>` (`a_iv` / `b_iv` on `diff`), `tensor`, `slice`, `slice.npy` and `diff` read an op in `saved` from
the intervention's tensors and an op outside `downstream` from the clean run (it is unchanged by definition); any
other op answers 404 `No intervened tensor saved for {op}: the capture budget ({mb} MB) was reached before it; the
graph still shows its Δ` or `…: the clean run saved no tensor for it`, and an unknown hash 404 `Intervention {hash}
not found in {run}`.

`tensor`, `slice`, `slice.npy` and `diff` answer 404 when an op has no tensor on disk, with a detail that says why:
`Op {op} not found in {run_id}` for an op outside the IR, else `No tensor saved for {op}: <reason>` where the reason
is `the capture budget ({mb} MB) was reached before this op; capture again with a larger budget`, `it is outside the
captured modules ({mods}); capture again with its module selected`, or `the op produced no tensor`.

## Errors (`errors.py`)

**Request guard** (`api.SameOrigin`, every route): a request whose `Host` is not this machine (localhost, an IP
literal, `NEUROTOME_HOST` or a name in `NEUROTOME_ALLOWED_HOSTS`) is refused with 403, so a page on another site
that rebinds its DNS name to 127.0.0.1 cannot read runs; a POST / PATCH / DELETE whose browser `Origin` is neither
the dev UI (`:5173`) nor this server gets 403 `Cross-origin request refused`, so a web page cannot upload files or
start captures (`py:` specs run code).

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

v0.3 messages (all 400; `interventions.normalise`, `RunManager.resolve` and the generation checks answer at POST,
the TorchLens and target checks in the job):

| Where | Message |
|---|---|
| the set | `An intervention set needs at least one edit` · `At most 32 edits per set` · `intervene jobs need interventions` |
| the op | `Unknown op {op}` · `{op} is the model's input or output; pick an op inside the model` · `{op} holds {dtype} values; only float tensors can be edited` |
| scope | `Channel {c} out of range for {op} ({n} channels)` · `{op} has shape {shape}: channel / position scopes need a (B, C, H, W), (B, T, D) or (B, D) tensor` · `{op} has no positions (shape {shape}); scope it by channels` · `Positions of {op} are [y, x] pixels (H {H}, W {W})` · `Positions of {op} are token indices below {T}` · `"last" applies to token positions (B, T, D)` |
| source runs | `Resample needs source_run (a run id or "random")` · `Source run {id} not found` · `{id} is a run of another model ({spec} · {weights}); use a run with the same model spec and weights` · `{id} has no saved tensor for {op}` · `{id}'s {op} has shape {a}, this run's {b}` · `No other run of this model has a saved tensor for {op}` · `Mean over runs needs another saved run of this model with a tensor for {op}` |
| steer | `Steering needs a direction` · `Steering needs a (B, C, H, W), (B, T, D) or (B, D) tensor; {op} is {shape}` · `Direction channel {c} out of range for {op} ({n} channels)` · `Contrast runs have {a} and {b} channels at {op}` |
| the job | `TorchLens cannot edit {op} ({func}): it returns its input unchanged in eval mode. Edit its parent {parent} instead.` (eval-mode dropout returns its input object, so TorchLens keeps the original value) · `The intervention changed the model's control flow; TorchLens cannot align the ops` · `{op} returns several tensors; edit one of its consumers` · `Target {t} is outside the output (0–{n-1})` |
| generate | `Generation needs a causal language model with a tokenizer; this run's model is not one.` · `Generation edits {op} through a forward hook on its module; {op} is not the output of a module (pick its module node)` · `Generation supports zero, mean over positions, constant, scale and steer edits ({kind} values have the prompt's length)` · `The prompt ({T} tokens) plus {n} new tokens exceeds the model's {P} positions` |

## Jobs (`jobs.py`)

Captures and attribution, layer, neuron, agreement, lens, intervene and generate requests run as jobs on **one** daemon worker thread fed by a
queue, so one job runs at a time. Jobs call `RunManager` methods, which take `RunManager._lock` themselves (the
worker never does; the lock is not re-entrant), so a synchronous capture and a job never share the GPU.
`RunManager.job()` validates the request, answers a cache hit with a job born `done`, and otherwise submits the
work keyed by its cache file, so a second identical request joins the queued or running job (never one already asked to cancel). Work reports
`progress(fraction, message)` (`Node gradients`, `IG batch i/n`, `Layer IG batch i/n`, `Conductance batch i/n`,
`Agreement · <method> (i/N)`, `Lens forward`, `Project layer i/L`, `Clean pass`, `Intervene`, `Save tensors i/N`,
`Clean i/n`, `Intervened i/n`); the fraction is monotone and held ≤ 0.99 until
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
Jaccard and overlap of the top 10% by |value| (exact zeros are never in a top set, so dead ReLU channels do not
count as agreement), and Spearman of average ranks (`null` for a constant vector). Results cached before 0.3.0
(without `seq_channels: "features"`) counted a token tensor's positions as its channels and are recomputed.

**Interventions** (`RunManager.intervene`, kind `intervene`) take an InterventionSet:

```jsonc
{ "target": null,                 // logit index reported as Δ (null = the clean run's top prediction)
  "edits": [                      // 1..32, one forward pass; several edits on one op compose in list order
    { "op": "conv2d_3_5",         // IR op id (a graph node's output_op); not an input/output op; float dtype
      "kind": "zero" | "mean" | "resample" | "constant" | "scale" | "steer",
      "channels": null | [0, 3],  // null = every channel
      "positions": null | "last" | [2, 5] | [[4, 7], [4, 8]],
      "over": "positions" | "runs",                 // mean
      "source_run": "run_0004" | "random", "seed": 0,  // resample
      "value": 0.0,                                 // constant
      "alpha": 1.0,                                 // scale (×α), steer (+α·v)
      "direction": {"kind": "contrast", "run_a", "run_b", "pool": "mean" | "last"} | {"kind": "channel", "channel": 3} } ] }
```

Scope axes: rank 4 `(B, C, H, W)` (attention `(B, H, Q, K)` too) has channels on axis 1 and `[[y, x], …]` pixel
positions; rank 3 `(B, T, D)` has features on axis 2 and token positions (`"last"` = T − 1, re-resolved per step in
generation); rank 2 `(B, D)` has channels only; other ranks take no scope. Every edit computes a full replacement
`r = f(x)` and writes `where(mask, r, x)`: `zero` 0, `mean` the per-channel mean over the position axes
(`positions`) or the element-wise mean of this op's saved tensor over every run of the same spec + weights and shape
(`runs`, ≥ 2; the runs on disk are the batch), `resample` the op's saved tensor in `source_run` (`random`:
`random.Random(seed).choice` over the other qualifying runs; an explicit source is labelled `Replace`), `constant`,
`scale` `α·x`, `steer` `x + α·v` with `v` = pooled(run B) − pooled(run A) (mean over positions, or the last token)
or the unit vector `e_c`, broadcast along the channel axis.

`interventions.normalise` returns the canonical set (only the keys of each kind, defaults filled, channels and
positions sorted and de-duplicated, `[]` → null); `hash = sha1(json.dumps(norm, sort_keys=True))[:12]`, so key and
list order never change it. `RunManager.resolve` reads the source runs (store only), then the job runs a clean pass
(`Clean pass`, the target), `downstream` (the edited ops and every IR descendant, IR order), `keep_ops` (the
downstream ops the clean run saved as floats, walked under the run's budget, `ir.model.budget_bytes` (0 is a
budget too) else the server's, with the never-cross rule: `truncated`), the backend's `intervene` (`Intervene`) and saves the kept
tensors (`Save tensors i/N`). Result (`meta.json`):

```jsonc
{ "hash", "run_id", "name": "Zero · block.conv2 · all (+1)", "created", "set": { /* normalised */ },
  "labels": ["Zero · block.conv2 · all", "Scale ×3 · block.relu@2 · ch 0–3"],   // <Kind> · <path> · <scope>
  "method_label": "Intervention · TorchLens capture-time hooks (every op)",   // NNsight: "Intervention · NNsight (module outputs)"
  "target": 3, "target_label": "cat", "before": {"logit", "prob"}, "after": {"logit", "prob"},
  "delta_logit", "delta_prob", "top_before": [/* TopK ×5 */], "top_after": [/* TopK ×5 */],
  "edited": [ops], "downstream": [ops], "saved": [ops],   // saved: a prefix of the eligible downstream ops
  "truncated": false, "budget_bytes": 2147483648, "bytes": 16384,
  "nodes": {"add_1_6": {"l2", "rel", "cos", "max_abs", "nonfinite"}},  // every downstream op with a float tensor
  "stats": {"add_1_6": {/* tensor_stats of the intervened tensor */}} }
```

A diff row is taken over positions where both tensors are finite: `l2 = ‖after − before‖₂`, `rel = l2 / ‖before‖`
(0 when ‖before‖ = 0), `cos` (null when a norm is 0), `max_abs`; `nonfinite` counts the positions where either is
NaN / inf and they differ. Non-finite logits and probs are null. No-ops by construction: `scale` α = 1, `steer`
α = 0, `resample` from this run.

**Generation** (`RunManager.generate`, kind `generate`, LM runs with `(1, T)` token ids) decodes greedily with
`model(ids)` per step (no KV cache; loaded models have `use_cache = False`), at most 64 tokens and never past the
model's positions, no early stop, once clean (`Clean i/n`) and once with the edits (`Intervened i/n`) as plain
forward hooks on the op's module site (`module_site`; a root pre-hook resets the per-module call counters each
forward, and the hook edits the first tensor of the output). Only zero, mean over positions, constant, scale and
steer apply (resample and runs-mean values have the prompt's length). Result (`generate_<n>.json`): `{hash, n,
labels, method_label, prompt: [decode([id]) per prompt id], clean: {ids, tokens, text, top: [[{id, token, prob}
×5] per step]}, steered: {…}, diverge: first differing step | null}`.

**Lens** (`RunManager.lens`, kind `lens`) runs `lens.logit_lens` on the live model and the run's `(1, T)` token
ids and caches `lens/<target|top>.json`.

## Capture backends: TorchLens vs NNsight

A run records its backend in `meta.model.backend` / `ir.model.backend`. `RunManager._backend(run_id)` imports that
module lazily, and attribution uses its `op_gradients` and interventions its `intervene`, so op ids always match the IR.
Choice at capture: an explicit `backend`, else **auto**: `nnsight` when the model has more parameters than
`NEUROTOME_NNSIGHT_PARAMS` (default 1e9) and nnsight is installed (the capture job then reports
`Large model (n B params) · NNsight module capture`), else `torchlens`.

| | TorchLens (default) | NNsight (`backend: nnsight`, auto above 1 B params) |
|---|---|---|
| Graph | every op, including functional ones (`add`, `softmax`, `view`) | leaf-module calls; functional ops between modules are invisible and edges fall back to call order |
| Attention maps | every softmax (not SDPA: torchvision `vit_b_16` fuses attention, so it has none) | only where a module outputs the probabilities (GPT-2 `attn_dropout`, which transformers 5 calls as a module only with `reorder_and_upcast_attn=True`); stock GPT-2, Llama and Qwen apply softmax and dropout functionally, so none |
| Merge shares | exact grad × input through merges | parent shares (approximate, labelled `share_method: parent`) |
| Interventions | every op (capture-time hooks) | leaf-module calls (NNsight `.output` edits) |
| Cost | records every op (memory and time grow with op count) | hooks + one float32 copy per kept leaf call; much smaller IR |

The logit lens and token attribution use plain hooks and Captum on the live model, so they work with either backend.

Interactive docs live at `http://127.0.0.1:7070/docs` while the server runs.

## Configuration and CLI

`NEUROTOME_RUNS_DIR` (default `./runs`), `NEUROTOME_HOST` / `NEUROTOME_PORT` (`127.0.0.1:7070`),
`NEUROTOME_DEVICE` (`cpu`), `NEUROTOME_MAX_CAPTURE_MB` (the per-run activation budget, default 2048; 0, a
negative value or `inf` means unlimited), `NEUROTOME_MAX_LIVE_MODELS` (default 2), `NEUROTOME_NNSIGHT_PARAMS`
(auto backend: NNsight above this many parameters, default 1e9) and `NEUROTOME_ALLOWED_HOSTS` (comma-separated
host names the server answers besides localhost and IP literals; see the request guard). A value that is not a
number stops the CLI with a one-line message.

- `neurotome serve [--host] [--port] [--runs-dir] [--device] [--reload]`
- `neurotome capture MODEL [--weights] [--image | --text "…" | --sample REF] [--backend torchlens|nnsight]
  [--runs-dir] [--modules layer3,layer4] [--max-capture-mb 512]`: synchronous; `--text` is a text sample (models
  with a tokenizer), `--sample` a dataset sample of a `py:` module (`load_sample(REF)`), `--backend` overrides the
  auto choice, `--modules` is comma-separated (whitespace stripped),
  `--max-capture-mb` overrides the budget for this run. A failed capture prints the plain message and exits 1.
- `neurotome.open(model, x, tokenizer=None)` captures in-process (synchronously) and serves the UI; a tokenizer
  labels token ids and, for a causal LM, enables the logit lens. A second call on the same port adds its run to
  the server that is already running.
- Extras: `pip install 'neurotome[hf,nnsight]'`.

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `lib/api.ts`, `lib/types.ts` | Typed client for the API above |
| `lib/router.ts` | Hash routes `#/runs/<id>/<graph\|tensors\|trace\|lens\|interventions>?node=…` (shareable deep links; the graph view also keeps `depth`, `expand`, `collapse`, `color`, and the intervention `iv` / `set`) |
| `lib/tokens.ts` | Token labels: `showToken` (visible whitespace), `tokenLabels(run, n)`, `spanLabel` for pooled cells, `tokenTone` for the token strip |
| `lib/lensState.ts` | Lens URL state (`t`, `mode`, `pos`), cell text, readout, `shortLayer` |
| `lib/attentionState.ts` | Attention view helpers: `headReadout`, `patternLabel`, `headColumns` (head-table columns, square vs not), `axisLetters` (`B H Q K`, `B Q H K`) |
| `lib/layout.ts` | ELK layered layout with nested groups. It runs in an ELK Web Worker (lazy-loaded; falls back to the bundled ELK where `Worker` is missing) behind a content-keyed cache, so pan, zoom, attribution and colour changes never re-lay out |
| `lib/viewport.ts` | Pure viewport maths: visible-node and edge culling, node/edge hit tests, keyboard `step`, colour `mix` |
| `lib/graphState.ts` | Pure graph-workspace state: fold (`expand`/`collapse`) URL params, search matching, colour modes (incl. `iv`: Intervention Δ; legacy `color=patch` parses as `iv`) |
| `lib/interventionState.ts` | The intervention draft in the URL (`parseSet` / `setParam`, `addEdit` / `removeEdit`), index lists (`0,3,7–9`), `scopeAxes` (§ Interventions), `editLabel` (the server's label words), `deltaColor` |
| `lib/composeState.ts` | Composing helpers: `toggleCells` / `cellsOf` (pixel selection through pooled cells), `divergeIndex`, `genReadout`, `sameModelRuns` |
| `lib/traceState.ts` | Trace view URL state (`m`, `t`, `obj`+`ch`, `steps`, `layer`, `lop`, `nc`, `go`), the attribution request, agreement cell tones, a series map's clock, row / column labels and block spans; a picked step's window rows / columns, `windowShares` (the "Where the attribution is" table), `zoomSpan`, `sliceText` / `blockChannels` / `seqTimeAxis` (the Values slice of a window or cell), `stepText` |
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
| `components/InputPanel.tsx`, `lib/inputState.ts` | The Input view (Tensors tab, input node): raw sample vs model input with pixel axis ranges, the preprocessing steps, per-channel value ranges; for a dataset sample the `Sample` info rows, the raw signal (one envelope plot per channel) and the (1, T, C) input as one heatmap per block, all on the raw signal's clock |
| `components/ValuesTable.tsx` | Slice expression box, the capped values table and the `.npy` download |
| `components/ComparePanel.tsx` | Compare two ops on a shared scale, with the diff statistics (attention ops show the selected head side by side) |
| `components/panels.tsx` | Module tree, node inspector (with `Trace from` channel link, `Layer attribution` link and the Intervene section) |
| `components/InterventionPanel.tsx` | Inspector section `Intervene · this node's output`: the node's Δ row under the open intervention, the edit form, the draft set and `Run set (n)` |
| `components/EditForm.tsx` | One edit: kind, kind parameters, channel and position scope (pixel picker on a DotMatrix, token selection on the TokenStrip), `Add to set` |
| `components/InterventionBanner.tsx` | The `Intervened` row under the TopBar (name, edits, Δ logit, saved / downstream tensors) with a `Clean` link (Graph, Tensor and Trace views) |
| `components/JobProgress.tsx` | Job progress line with `Cancel`; the error text of a failed job |
| `components/CaptureBanner.tsx` | The `Truncated` / `Selective` row under the TopBar for partial captures (Graph and Tensor views) |
| `components/AgreementMatrix.tsx` | n×n method-agreement matrix (Jaccard or Spearman) with a hover readout and PNG export |
| `components/SeriesMap.tsx` | A sequence input's attribution (Trace view): channel × time heatmap with block separators and names, the signed Σ per time column, block shares; with a picked step its window bands (`Band`, `StepLine`, `WindowLegend`, shared with the Input view), the `Full` / `Around step` zoom, the "Where the attribution is" table and the `Show input values` / `Show on input` links |
| `components/TokenStrip.tsx` | Token attribution as inline tokens tinted by score (group `Token attribution`), with a readout |
| `views/InterventionsView.tsx` | Interventions tab: compose a set (target, edits, add-edit form), run it, the saved list (open, rename, delete), the result (Δ logit / prob, top-k before / after, links) and generation (clean vs intervened continuations, per-step top tokens) |
| `views/LensView.tsx` | Logit lens grid (layers × positions, `Top-1` / `Target` colour), target across layers, top-k at a cell |
| `views/*` | Home (text / file sample, capture backend), Graph, Tensors (incl. the attention view), Trace (incl. the token strip), Lens |
| `styles/tokens.css` | Design tokens (mirrors `design/tokens.json`) |

Tensor view URL: `#/runs/<id>/tensors?node=…&c=…&py=…&px=…&norm=global&cmap=div&cmp=…&iv=…&vs=clean` (defaults omitted;
`cmp` turns on compare mode; `iv` reads every tensor through that intervention; `vs=clean`, only with `iv`, compares
the clean (A) and intervened (B) tensor of the selected node). Overlay on/off and opacity are view state, not URL
state.

Graph URL: `#/runs/<id>/graph?node=…&depth=…&expand=…&collapse=…&color=plain|attr|norm|sparsity|nan|iv&iv=<hash>&set=<JSON
InterventionSet>` (`color=iv` is the Intervention Δ mode; legacy `color=patch` parses as `iv`; legacy `patch` / `pb`
/ `psrc` are ignored and dropped). `iv` is the open intervention (Graph, Tensors, Trace, Interventions); `set` is the
draft being composed (Graph, Interventions). TopBar tab links carry `iv` (not to Lens) and `set` (to Graph and
Interventions) next to `node`.

Interventions URL: `#/runs/<id>/interventions?node=<op or node id>&iv=<hash>&set=<JSON>` (`node` is the op the
Add-edit form targets).

Trace view URL: `#/runs/<id>/trace?node=…&m=ig&t=207&obj=<op>&ch=<c>&steps=32&layer=layer_ig&lop=<op>&nc=<c>&go=1&iv=…`
(defaults omitted; with `iv`, contributor rows show the intervention's relative Δ; attribution stays on the clean pass). `go=1` runs the attribution, then the layer and neuron jobs, once and drops itself; the
Tensor view and the graph inspector link here with it.

Lens URL: `#/runs/<id>/lens?t=<token id>&mode=target&pos=<position>` (defaults omitted: `t` = the top prediction,
`mode` = `top`, `pos` = the last position). The Lens tab is disabled for runs whose `model.lm` is not true.

Dev: Vite on :5173 proxies `/api` to :7070. Build: output goes to `backend/neurotome/static/`, so the Python package serves the UI;
`index.html` is served with `Cache-Control: no-cache` so a rebuilt UI never points at deleted hashed assets. Every other path
falls back to `index.html`, except `/api/*` (a JSON 404) and paths that resolve outside `static/` (never served).

## Extension points

- **New capture backend** (`torch.export`, forward hooks; `capture_nnsight.py` is the worked example): implement
  `capture()`, `op_gradients()` (including the `merges` field and the unit objective) and `intervene()`, returning
  the same IR, score and diff dicts, and name it in `runs._module` / `BACKENDS`. Nothing downstream changes. The
  intervention contract:

  ```python
  Edit = tuple[str, Callable[[torch.Tensor], torch.Tensor]]   # (op id, fn: the op's output -> same-shape tensor)

  def intervene(model, inputs, edits: list[Edit], downstream: Collection[str], keep: Collection[str] = ()) -> dict:
      """One forward pass with every edit applied where its op runs; edits on one op compose in list order.
      Returns {"nodes": {op: diff row}, "stats": {op: tensor_stats(after)}   # downstream ops with a float tensor
               "after": {op: np.ndarray float32}                           # op in keep (CPU copies)
               "logits_after": the model output (anything models.to_logits accepts)}.
      ValueError: "Unknown op …", an fn that changes the shape ("Edit of {op} changed its shape {a} -> {b}"),
      an op the backend cannot edit."""
  ```

  Edit functions come from `interventions.edit_fn` (pure torch: the mask is rebuilt from each call's shape, values
  are cast to the tensor's dtype and device on first use, and the result always owns its memory so an in-place op
  after the edit is safe); the backend only decides *where* they run. `intervene` runs with grad enabled, as the
  capture did: under `no_grad`, `nn.MultiheadAttention` takes a fused fast path and the op ids stop matching the IR.
- **New edit kind**: a branch in `interventions.normalise` (its keys and checks), `edit_fn` (its replacement) and
  `label`, a value in `RunManager.resolve` if it reads other runs, the kind in `api.EditIn`, and the client's
  `EditKind` / `editLabel` / `EditForm`.
- **A time-series model with a dataset**: give its `py:` module `list_samples()` / `load_sample(ref)` (and
  `OUTPUT_KIND = "values"` for score outputs); see `examples/signal_model.py` and the README. Neurotome needs no
  change: the Home form offers the samples, the Input view draws `raw`, `blocks` and `time` generically.
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
