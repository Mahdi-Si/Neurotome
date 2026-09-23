# Neurotome

**Open the model. See every signal.** A local web workbench for dissecting one PyTorch model on one
input. It captures every op of a forward pass, lets you browse the computation graph, and opens any
tensor as maps, signals or projections. It also traces which parts of the model and the input drove an output.

![Graph workspace](docs/screenshots/graph.png)

| Tensors | Trace |
|---|---|
| ![Tensor inspector](docs/screenshots/tensors.png) | ![Attribution trace](docs/screenshots/trace.png) |

> Status: **v0.1 starter.** The full pipeline works end to end. See [docs/PLAN.md](docs/PLAN.md) for
> the MVP scope, roadmap and step-by-step implementation plan.

## Quick start

Requirements: Python 3.10+, Node 20+.

```bash
make install        # Python venv + editable backend + npm packages
make dev            # API on :7070 and UI on :5173 (hot reload)
# open http://localhost:5173 → "Capture forward pass"
```

Without `make`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "./backend[dev]"
cd frontend && npm install && cd ..
neurotome serve --reload &          # API :7070
cd frontend && npm run dev          # UI  :5173
```

**GPU / CPU-only PyTorch.** `pip` installs the default PyTorch build for your platform. For a specific
CUDA version or a small CPU-only build, install torch first by following https://pytorch.org, then run `make install`.

### Windows

`make` is optional; with [uv](https://docs.astral.sh/uv/) and PowerShell:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -e "./backend[dev]"
cd frontend; npm ci; cd ..
.\scripts\dev.ps1                   # API :7070 (reload) + UI :5173; Ctrl+C stops both
```

The Makefile also works with a Windows venv (`.venv/Scripts`) if you have `make` from Git Bash or MSYS2.

### From Python or a notebook

```python
import neurotome
neurotome.open(model, x)               # captures, then serves the UI at http://127.0.0.1:7070
neurotome.open(model, x, block=False)  # notebooks: server runs in the background
```

### From the command line

```bash
neurotome capture torchvision:resnet18 --weights DEFAULT --image cat.jpg
neurotome capture py:examples.custom_model:build   # your own model (see examples/custom_model.py)
neurotome serve                                    # serves the built UI + API on :7070
```

### Single-process build

```bash
make build          # builds the UI into backend/neurotome/static
neurotome serve     # UI and API together on http://127.0.0.1:7070
```

## Graph workspace

- **Fold per module**: `+` on a module node expands it, `−` on a group box collapses it; the depth
  buttons reset these overrides. The fold lives in the URL (`#/runs/<id>/graph?depth=1&expand=layer1.0:1&collapse=…`),
  so a link reopens the same view.
- **Search** (`/` or Ctrl/Cmd+K): name, op type or shape (`64x56x56`). Picking a hit that is folded
  inside a collapsed module opens its ancestors, selects it and centres it.
- **Keyboard** (click or Tab into the canvas): ↓ child, ↑ parent, ←/→ siblings, `Enter` opens the
  tensor, `Escape` clears the search, `+`/`-` expand/collapse the selection.
- **Colour**: Plain · Attr (grad × act) · Norm (RMS) · Sparsity (zero %) · NaN/Inf flag, with a legend.
- **Hover** an edge for the shape it carries and any dtype/device change; the minimap (bottom left)
  pans on click or drag.

## Layout

```
backend/            Python package `neurotome` (FastAPI, TorchLens, Captum)
  neurotome/        capture.py · graph.py · tensors.py · attribution.py · runs.py · api.py · cli.py
  tests/            end-to-end tests (tiny model + ResNet-18)
frontend/           Vite + React + TypeScript UI
  src/views/        Home · Graph · Tensors · Trace
  src/components/   GraphCanvas (ELK worker, canvas edges), Minimap, GraphSearch, charts, panels
  src/lib/          pure logic: layout, viewport, graphState (tested with vitest)
design/             tokens.json + the canvas mockups that set the visual direction
docs/               PLAN.md (MVP + roadmap + steps) · ARCHITECTURE.md · DESIGN.md · screenshots
examples/           quickstart.py · custom_model.py
```

## Commands

| | |
|---|---|
| `make test` | backend pytest + frontend typecheck and vitest |
| `make build` | production UI into the Python package |
| `make serve` | run the packaged server |
| `make clean` | remove builds, caches and local runs |

## Configuration

| Env var | Default | |
|---|---|---|
| `NEUROTOME_RUNS_DIR` | `./runs` | where runs are stored |
| `NEUROTOME_HOST` / `NEUROTOME_PORT` | `127.0.0.1` / `7070` | keep localhost: `py:` model specs import and run code |
| `NEUROTOME_DEVICE` | `cpu` | e.g. `cuda`, `cuda:1`, `mps` |
| `NEUROTOME_MAX_CAPTURE_MB` | `2048` | activation budget per run; later ops are recorded without tensors |
| `NEUROTOME_MAX_LIVE_MODELS` | `2` | models (+ inputs) kept loaded for attribution; least recently used is evicted |

## Built on

[TorchLens](https://github.com/johnmarktaylor91/torchlens) (capture) · [Captum](https://github.com/meta-pytorch/captum)
(attribution) · [ELK](https://eclipse.dev/elk/) via elkjs (layout) · FastAPI · React · Geist fonts.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Mahdi-Si.
