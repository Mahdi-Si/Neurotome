# Neurotome

Open a PyTorch model and see every signal: the computation graph, every activation and attribution, in one
local web UI.

Neurotome runs one forward pass through TorchLens, stores every
op's output on disk and serves a browser workspace on `localhost`:

- **Graph**: the whole computation graph, folded by module depth, searchable, coloured by activation statistics or
  attribution, with the dominant path from the output back to the input.
- **Tensors**: any op's output as channel maps, sequences or vectors, slices as tables or `.npy` downloads, and a
  side-by-side diff of two ops.
- **Trace**: integrated gradients, saliency and gradient × input on the input (per token on the embeddings for
  text); gradient × activation on every op; layer and neuron conductance; method agreement. Every number names the
  method behind it.
- **Interventions**: zero, mean, resample, constant, scale and steering edits on any op's output, scoped by channels
  and positions, run in one forward pass with a before/after diff of every downstream node; for causal LMs, clean vs
  intervened generation side by side.
- **Sequences and LLMs**: Hugging Face models (`hf:gpt2`), text samples, per-head attention maps and the logit lens;
  an NNsight backend captures module outputs of models too big to record op by op.

Tensors never leave your machine: the browser only receives small server-side summaries.

## Install

```bash
pip install neurotome
```

Requires Python 3.10+ and PyTorch 2.2+ (install the CUDA build of PyTorch first if you want to capture on a GPU).
Optional extras: `hf` for Hugging Face models and `nnsight` for the NNsight capture backend:

```bash
pip install "neurotome[hf,nnsight]"
```

## Use

Start the server and open the UI at <http://127.0.0.1:7070>:

```bash
neurotome serve                      # --device cuda, --port, --runs-dir
```

Pick a model (`torchvision:resnet18`, `hf:gpt2`, or `py:package.module:factory` for your own), a sample and press
**Capture forward pass**. Or capture from the command line:

```bash
neurotome capture torchvision:resnet50 --weights DEFAULT --image cat.jpg
neurotome capture hf:gpt2 --weights DEFAULT --text "The cat sat on the"
neurotome capture py:mypkg.models:build --modules layer3,layer4 --max-capture-mb 512
neurotome capture py:mypkg.models:build --sample rec01:0   # a dataset sample: the module's load_sample(ref)
```

From a script or notebook:

```python
import neurotome

neurotome.open(model, x)               # capture model(x), then serve the UI
neurotome.open(model, x, block=False)  # notebooks: serve in the background
neurotome.open(lm, ids, tokenizer=tok) # token labels and, for a causal LM, the logit lens
```

Runs live in `./runs` (`NEUROTOME_RUNS_DIR`). Other settings: `NEUROTOME_DEVICE`, `NEUROTOME_PORT`,
`NEUROTOME_HOST`, `NEUROTOME_MAX_CAPTURE_MB` (the per-run disk budget for activations),
`NEUROTOME_MAX_LIVE_MODELS` and `NEUROTOME_NNSIGHT_PARAMS` (NNsight capture above this many parameters).

A `py:` module may also define `INPUT_SHAPE`, `CATEGORIES`, `TOKENIZER`, `preprocess(img)`, `OUTPUT_KIND = "values"`
(score outputs: no softmax) and a dataset, `list_samples()` / `load_sample(ref)`; see `examples/custom_model.py`
and `examples/signal_model.py`. Put another repository's modules on `PYTHONPATH` to load them.

`py:` specs import and run code, so the server binds to localhost by default; do not expose it on a network.

## Documentation

See the Neurotome documentation site (Getting started, Architecture, Design) for the full guide.

## License

Apache-2.0
