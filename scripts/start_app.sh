#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SIGNBRIDGE_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

BACKEND_PORT="${SIGNBRIDGE_BACKEND_PORT:-8001}"
FRONTEND_PORT="${PORT:-${SIGNBRIDGE_FRONTEND_PORT:-3099}}"
FRONTEND_HOST="${SIGNBRIDGE_HOST:-127.0.0.1}"

export SIGNBRIDGE_API_URL="http://127.0.0.1:${BACKEND_PORT}"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data/personalization"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting SignBridge FastAPI backend on 127.0.0.1:${BACKEND_PORT}..."
"$PYTHON_BIN" -m uvicorn signbridge.web.app:app \
  --host 127.0.0.1 \
  --port "$BACKEND_PORT" \
  >"$ROOT_DIR/logs/backend.log" 2>&1 &
BACKEND_PID=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
  echo "Backend failed to become healthy on port ${BACKEND_PORT}. Check logs/backend.log" >&2
  exit 1
fi

echo "Starting SignBridge Next.js frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}..."
cd "$ROOT_DIR/signbridge-web"
if [[ -x "./node_modules/.bin/next" ]]; then
  ./node_modules/.bin/next start -H "$FRONTEND_HOST" -p "$FRONTEND_PORT" \
    >"$ROOT_DIR/logs/frontend.log" 2>&1 &
else
  npx next start -H "$FRONTEND_HOST" -p "$FRONTEND_PORT" \
    >"$ROOT_DIR/logs/frontend.log" 2>&1 &
fi
FRONTEND_PID=$!

wait "$FRONTEND_PID"
