# Dissect

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
dissect serve --reload &            # API :7070
cd frontend && npm run dev          # UI  :5173
```

**GPU / CPU-only PyTorch.** `pip` installs the default PyTorch build for your platform. For a specific
CUDA version or a small CPU-only build, install torch first by following https://pytorch.org, then run `make install`.

### From Python or a notebook

```python
import dissect
dissect.open(model, x)            # captures, then serves the UI at http://127.0.0.1:7070
dissect.open(model, x, block=False)  # notebooks: server runs in the background
```

### From the command line

```bash
dissect capture torchvision:resnet18 --weights DEFAULT --image cat.jpg
dissect capture py:examples.custom_model:build     # your own model (see examples/custom_model.py)
dissect serve                                      # serves the built UI + API on :7070
```

### Single-process build

```bash
make build          # builds the UI into backend/dissect/static
dissect serve       # UI and API together on http://127.0.0.1:7070
```

## Layout

```
backend/            Python package `dissect` (FastAPI, TorchLens, Captum)
  dissect/          capture.py · graph.py · tensors.py · attribution.py · runs.py · api.py · cli.py
  tests/            end-to-end tests (tiny model + ResNet-18)
frontend/           Vite + React + TypeScript UI
  src/views/        Home · Graph · Tensors · Trace
  src/components/   GraphCanvas (ELK), charts, panels
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
| `DISSECT_RUNS_DIR` | `./runs` | where runs are stored |
| `DISSECT_HOST` / `DISSECT_PORT` | `127.0.0.1` / `7070` | keep localhost: `py:` model specs import and run code |
| `DISSECT_DEVICE` | `cpu` | e.g. `cuda`, `cuda:1`, `mps` |
| `DISSECT_MAX_CAPTURE_MB` | `2048` | activation budget per run; later ops are recorded without tensors |

## Built on

[TorchLens](https://github.com/johnmarktaylor91/torchlens) (capture) · [Captum](https://github.com/meta-pytorch/captum)
(attribution) · [ELK](https://eclipse.dev/elk/) via elkjs (layout) · FastAPI · React · Geist fonts.
