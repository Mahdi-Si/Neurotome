VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install dev build serve test test-backend test-frontend lint clean

$(VENV)/bin/python:
	python3 -m venv $(VENV)

install: $(VENV)/bin/python
	$(PIP) install -U pip
	$(PIP) install -e "./backend[dev]"
	cd frontend && npm install

dev:
	./scripts/dev.sh

build:
	cd frontend && npm run build

serve:
	$(VENV)/bin/dissect serve

test: test-backend test-frontend

test-backend:
	cd backend && ../$(PY) -m pytest -q

test-frontend:
	cd frontend && npm run typecheck && npm test

lint:
	cd backend && ../$(PY) -m ruff check .

clean:
	rm -rf backend/dissect/static frontend/dist runs .pytest_cache backend/.pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
