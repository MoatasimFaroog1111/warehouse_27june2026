#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/GroundingDINO/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "GroundingDINO virtual environment was not found at ${VENV_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"

ENV_FILE_ARGS=()
if [[ -f ".env" ]]; then
  ENV_FILE_ARGS=(--env-file .env)
fi

exec python -m uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WEB_CONCURRENCY" \
  --log-level "${LOG_LEVEL,,}" \
  --proxy-headers \
  --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}" \
  --no-access-log \
  "${ENV_FILE_ARGS[@]}"
