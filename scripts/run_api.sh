#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/GroundingDINO/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "GroundingDINO virtual environment was not found at ${VENV_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
