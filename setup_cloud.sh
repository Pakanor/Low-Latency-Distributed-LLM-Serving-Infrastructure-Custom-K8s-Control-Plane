#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON="${VENV_DIR}/bin/python"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.txt"
"${PYTHON}" -m pip install -e "${ROOT_DIR}"

PYTHONPATH="${ROOT_DIR}" "${PYTHON}" -c '
from huggingface_hub import snapshot_download
from app.config import DEFAULT_CONFIG
snapshot_download(DEFAULT_CONFIG.model_name)
'

mkdir -p "${ROOT_DIR}/results/cloud-smoke"
HF_HUB_OFFLINE=1 PYTHONPATH="${ROOT_DIR}" "${PYTHON}" \
  "${ROOT_DIR}/run_suite.py" --profile smoke \
  --results-dir "${ROOT_DIR}/results/cloud-smoke"
