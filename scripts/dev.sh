#!/usr/bin/env bash
# Run the API (:7070, auto-reload) and the UI dev server (:5173) together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

DISSECT=".venv/bin/dissect"
[ -x "$DISSECT" ] || DISSECT="dissect"

"$DISSECT" serve --reload &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

(cd frontend && npm run dev)
