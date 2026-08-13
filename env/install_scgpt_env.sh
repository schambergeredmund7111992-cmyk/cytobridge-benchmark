#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENVS_ROOT="${VENVS_ROOT:-/tmp/${USER}/venvs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/${USER}/pip-cache}"
TMPDIR="${TMPDIR:-/tmp/${USER}/tmp}"
SCGPT_ENV="${SCGPT_ENV:-$VENVS_ROOT/cytobridge-scgpt-py310}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export PIP_CACHE_DIR TMPDIR

if [[ -z "${BASE_PYTHON:-}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3.10)"
  elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
  else
    echo "ERROR: set BASE_PYTHON to a Python 3.10 executable." >&2
    exit 2
  fi
fi

mkdir -p "$VENVS_ROOT" "$PIP_CACHE_DIR" "$TMPDIR"

if [[ ! -x "$SCGPT_ENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$SCGPT_ENV"
fi

"$SCGPT_ENV/bin/python" -m pip install -U pip wheel
"$SCGPT_ENV/bin/python" -m pip install setuptools==70.3.0
"$SCGPT_ENV/bin/python" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121
"$SCGPT_ENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-scgpt.txt"
"$SCGPT_ENV/bin/python" -m pip install -e "$PROJECT_DIR" --no-deps

"$SCGPT_ENV/bin/python" - <<'PY'
import sys
import torch
import torchtext
import scgpt

print("scgpt env ok")
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchtext", torchtext.__version__)
print("scgpt", getattr(scgpt, "__version__", "unknown"))
PY
