# Release checklist

How a version of Neurotome gets from this repository to PyPI. The steps run **in this order**, and nothing is
published before the maintainer's explicit green light in step 4. The workflows (`.github/workflows/ci.yml`,
`release.yml`, `docs.yml`) are written and checked locally; none of them has run yet.

## 1. Push to GitHub

- Create the GitHub repository and push `main`.
- Then add the links that need its URL: `repo_url` in `mkdocs.yml`, and `[project.urls]` (Homepage, Documentation,
  Issues) in `backend/pyproject.toml`.

## 2. CI green

The `ci` workflow runs on every push to `main` and on pull requests. All four jobs must pass:

| Job | Checks |
|---|---|
| `backend` | `pytest -q`, `ruff check .`, `pyright backend/neurotome` (CPU torch, Python 3.11) |
| `frontend` | `npm run typecheck`, `npm test`, `npm run build` |
| `wheel` | builds the UI, `python -m build backend`, `scripts/check_wheel.py` on the wheel and sdist; uploads `dist` |
| `e2e` | CPU torch, `npm run e2e` (Playwright, Chromium); uploads the report on failure |

## 3. Maintainer manual testing and demo sign-off

Local pre-flight from the repository root (a `.venv` with `./backend[dev,docs]` installed):

```bash
cd frontend && npm run build && cd ..                        # UI into backend/neurotome/static
uv build backend                                             # backend/dist/neurotome-<v>.tar.gz + .whl
python scripts/check_wheel.py "backend/dist/*.whl" --sdist "backend/dist/*.tar.gz" --install
mkdocs build --strict                                        # docs site into site/
python scripts/bench.py                                      # success criteria, exits 1 on a miss
cd frontend && npm run e2e                                   # browser end-to-end suite
```

Then try, in the browser, on a clean runs dir:

- [ ] a torchvision model with pretrained weights and an uploaded image (`torchvision:resnet50`, `DEFAULT`, a photo);
      the top prediction on Home makes sense
- [ ] a `py:` model (`py:examples.custom_model:build`, and one of your own)
- [ ] selective capture (`Modules` = `layer3`): the `Selective` banner, `∅` nodes outside it, their inspector text
- [ ] a budget-truncated run (`Budget · MB` = 20 on ResNet-50): the `Truncated` banner and the saved-op count
- [ ] cancel a capture while it runs; a bad spec shows a readable error
- [ ] each view: Graph (fold, search, keyboard, colour modes, minimap), Tensors (pixel pick, slices, compare,
      PNG / `.npy` export), Trace (IG with completeness Δ, a unit objective)
- [ ] the jobs: IG, layer IG and conductance, neuron conductance, method agreement, patch & re-run, each with
      progress and `Cancel`
- [ ] dark theme (toggle, reload keeps it, charts and canvases redraw), and a quick look with Windows high contrast
- [ ] the wheel installed in a clean venv: `pip install backend/dist/*.whl`, `neurotome serve`, capture a run
- [ ] `neurotome.open(model, x, block=False)` from a notebook

Sign-off is the maintainer's explicit note (issue comment or chat) that the demo passed.

## 4. Tag → PyPI (only after the maintainer's explicit green light)

1. Bump `neurotome.__version__` in `backend/neurotome/__init__.py` (the only version source; the package version is
   dynamic) and commit.
2. One-time PyPI setup: on pypi.org add a **trusted publisher** (pending publisher for a new project) with project
   `neurotome`, owner and repository of the GitHub repo, workflow `release.yml`, environment `pypi`. No API token
   is stored anywhere.
3. One-time GitHub setup: create the environment `pypi` (Settings → Environments) with **required reviewers**; this
   is the manual approval gate of the publish job.
4. Tag and push:

    ```bash
    git tag vX.Y.Z && git push origin vX.Y.Z
    ```

    `release.yml` fails unless the tag equals `v` + `__version__`, builds the UI, the sdist and the wheel, runs
    `check_wheel.py`, and uploads `dist`.
5. Approve the `publish` job in the Actions run; it uploads to PyPI through trusted publishing.
6. Docs: the same tag triggers `docs.yml`, which deploys the versioned site with `mike` to the `gh-pages` branch
   (`X.Y.Z`, alias `latest`, default `latest`). Enable GitHub Pages on `gh-pages` once. `workflow_dispatch` redeploys.
7. Check `pip install neurotome==X.Y.Z` in a clean venv.

## Recorded local checks

**Wheel check, 23 September 2026** (Windows 11, Python 3.12.1, uv 0.11.29, Node 24):

```text
$ cd frontend && npm run build
$ uv build backend
Successfully built backend\dist\neurotome-0.1.0.tar.gz
Successfully built backend\dist\neurotome-0.1.0-py3-none-any.whl
$ python scripts/check_wheel.py "backend/dist/*.whl" --sdist "backend/dist/*.tar.gz" --install
ok    wheel neurotome-0.1.0-py3-none-any.whl
ok    sdist neurotome-0.1.0.tar.gz
ok    install + serve (clean venv, CPU torch)
```

The install step made a fresh uv venv in a temp dir, installed CPU torch and then the wheel, ran
`neurotome --help`, started `python -m neurotome.cli serve --device cpu` from the temp dir, and got `/api/health`
ok, `GET /` 200 with the root div and one bundled asset 200 (about 50 s in total). Benchmarks are recorded in
[Plan §5](PLAN.md#success-criteria).
