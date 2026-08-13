#!/usr/bin/env bash
# Build both protocol cache files from the same frozen drug-only representation.
set -euo pipefail

usage() {
  echo "usage: $0 --gpu INDEX --python PATH" >&2
}

GPU=""
PYTHON_BIN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
if [[ -z "$GPU" || -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  usage
  exit 2
fi
PYTHON_PREFIX="$(cd "$(dirname "$(readlink -f "$PYTHON_BIN")")/.." && pwd)"
if [[ "${CYTOBRIDGE_LAUNCH_CONFIRMED:-0}" != "1" ]]; then
  echo "cache launch refused: resource estimate has not been explicitly confirmed" >&2
  exit 3
fi

IFS=',' read -r UTIL MEMORY TOTAL < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
)
UTIL="${UTIL//[[:space:]]/}"
MEMORY="${MEMORY//[[:space:]]/}"
if (( UTIL >= 5 || MEMORY >= 1024 )); then
  echo "GPU $GPU is not eligible: utilization=${UTIL}%, memory=${MEMORY}/${TOTAL} MiB" >&2
  exit 4
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for protocol in drug_disjoint_v2 scaffold_disjoint_v2; do
  output="data/cache/sciplex_accept/$protocol/molformer_emb.npz"
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PYTHON_BIN" -m cytobridge.encoders.molformer_wrapper \
    --smiles_csv data/raw/sciplex/sciplex3_drugs.csv \
    --out "$output" \
    --device cuda \
    --batch-size 64
done

if [[ -x "$HOME/scripts/notify.py" ]]; then
  "$HOME/scripts/notify.py" --title "CytoBridge MolFormer" \
    --text "both frozen sci-Plex MolFormer caches passed" --level info || true
fi
