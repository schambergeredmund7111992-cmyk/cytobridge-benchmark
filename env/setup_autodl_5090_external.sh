#!/usr/bin/env bash
# Build the pinned chemCPA/biolord sources on a Blackwell-compatible torch runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
EXTERNAL_ENV="${EXTERNAL_ENV:-$HOME/miniconda3/envs/cytobridge-accept-5090-external}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$HOME/autodl-tmp/cytobridge-accept-external}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "missing conda executable: $CONDA_BIN" >&2
  exit 2
fi

if [[ ! -x "$EXTERNAL_ENV/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$EXTERNAL_ENV" \
    python=3.10 pip 'libstdcxx-ng>=12'
fi

PYTHON="$EXTERNAL_ENV/bin/python"
"$PYTHON" -m pip install pip==24.0 setuptools==70.3.0 wheel==0.43.0
"$PYTHON" -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$ROOT/env/requirements-autodl-5090-external.txt"

CONDA_BIN="$CONDA_BIN" \
EXTERNAL_ENV="$EXTERNAL_ENV" \
EXTERNAL_ROOT="$EXTERNAL_ROOT" \
  bash "$ROOT/env/setup_external_baselines.sh"
"$PYTHON" -m pip check

VERIFY_ARGS=(
  --expected-torch 2.7.1
  --required-cuda-prefix 12.8
  --out "$EXTERNAL_ROOT/state/runtime-verification.json"
)
if [[ "${CYTOBRIDGE_REQUIRE_BLACKWELL:-1}" == "1" ]]; then
  VERIFY_ARGS+=(--require-blackwell)
else
  VERIFY_ARGS+=(--no-require-blackwell)
fi
PYTHONPATH="$ROOT" "$PYTHON" -m scripts.verify_gpu_runtime "${VERIFY_ARGS[@]}"

echo "AutoDL RTX 5090 external environment ready: $EXTERNAL_ENV"
