#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_NAME="nasa"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if ! conda env list | grep -q "\b${ENV_NAME}\b"; then
  conda env create -f backend/environment.yml -n "$ENV_NAME"
else
  conda env update -f backend/environment.yml -n "$ENV_NAME"
fi

conda run -n "$ENV_NAME" python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend
