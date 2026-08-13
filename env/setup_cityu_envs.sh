#!/usr/bin/env bash
# Build only the core and scGPT acceptance environments on cityu.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CORE_ENV="${CORE_ENV:-$HOME/anaconda3/envs/cytobridge-accept-core}"
SCGPT_ENV="${SCGPT_ENV:-$ROOT/../.venvs/cytobridge-accept-scgpt}"

finish() {
  local rc=$?
  trap - EXIT
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge environment" \
      --text "cityu environment setup exit=${rc}" || \
      echo "environment completion notification failed" >&2
  fi
  exit "$rc"
}
trap finish EXIT

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "missing cityu conda executable: $CONDA_BIN" >&2
  exit 2
fi

if [[ ! -x "$CORE_ENV/bin/python" ]]; then
  # The cityu base conda has an optional authentication plugin whose unset host
  # causes an ``aau_token_host`` write failure in non-interactive sessions.
  # Disable plugins and select the built-in classic solver explicitly.
  env CONDA_NO_PLUGINS=true "$CONDA_BIN" env create --solver classic \
    -p "$CORE_ENV" -f "$ROOT/env/environment-accept-core.yml"
fi
"$CORE_ENV/bin/python" -m pip install --no-deps setuptools==70.3.0
"$CORE_ENV/bin/python" -m pip install -e "$ROOT" --no-deps

BASE_PYTHON="$CORE_ENV/bin/python" \
SCGPT_ENV="$SCGPT_ENV" \
VENVS_ROOT="$(dirname "$SCGPT_ENV")" \
PIP_CACHE_DIR="/tmp/${USER}/cytobridge-pip-cache" \
TMPDIR="/tmp/${USER}/cytobridge-tmp" \
bash "$ROOT/env/install_scgpt_env.sh"

env LD_LIBRARY_PATH="$CORE_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CORE_ENV/bin/python" -c \
  'import torch, pytorch_lightning, scanpy, rdkit, pyarrow; print("core ok", torch.__version__, torch.cuda.is_available())'
"$SCGPT_ENV/bin/python" -c \
  'import torch, torchtext, scgpt; print("scgpt ok", torch.__version__, torch.cuda.is_available())'
