VENV ?= .venv
# POSIX venvs put executables in bin/, Windows venvs in Scripts/.
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
EXE := .exe
PYTHON ?= python
DEV := powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev.ps1
else
BIN := $(VENV)/bin
PYTHON ?= python3
DEV := ./scripts/dev.sh
endif
PY := $(BIN)/python
PIP := $(PY) -m pip

.PHONY: install dev build serve test test-backend test-frontend lint clean

$(PY)$(EXE):
	$(PYTHON) -m venv $(VENV)

install: $(PY)$(EXE)
	$(PIP) install -U pip
	$(PIP) install -e "./backend[dev]"
	cd frontend && npm install

dev:
	$(DEV)

build:
	cd frontend && npm run build

serve:
	$(BIN)/neurotome serve

test: test-backend test-frontend

test-backend:
	cd backend && ../$(PY) -m pytest -q

test-frontend:
	cd frontend && npm run typecheck && npm test

lint:
	cd backend && ../$(PY) -m ruff check .

clean:
	rm -rf backend/neurotome/static frontend/dist runs .pytest_cache backend/.pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
