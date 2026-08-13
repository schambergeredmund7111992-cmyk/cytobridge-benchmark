#!/usr/bin/env bash
# Build an isolated Blackwell-compatible core environment without scGPT/torchtext.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/miniconda3/envs/cytobridge-accept-5090}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "missing conda executable: $CONDA_BIN" >&2
  exit 2
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" \
    python=3.10 pip 'libstdcxx-ng>=12'
fi

PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -m pip install pip==24.0 setuptools==70.3.0 wheel==0.43.0
"$PYTHON" -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$ROOT/env/requirements-autodl-5090-core.txt"
"$PYTHON" -m pip install -e "$ROOT" --no-deps
"$PYTHON" -m pip check

VERIFY_ARGS=(
  --expected-torch 2.7.1 \
  --required-cuda-prefix 12.8 \
  --out "$ENV_PREFIX/runtime-verification.json"
)
if [[ "${CYTOBRIDGE_REQUIRE_BLACKWELL:-1}" == "1" ]]; then
  VERIFY_ARGS+=(--require-blackwell)
else
  VERIFY_ARGS+=(--no-require-blackwell)
fi
"$PYTHON" -m scripts.verify_gpu_runtime "${VERIFY_ARGS[@]}"

echo "AutoDL RTX 5090 core environment ready: $ENV_PREFIX"
