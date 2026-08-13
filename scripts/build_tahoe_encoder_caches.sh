#!/usr/bin/env bash
# Build protocol-aligned Tahoe scGPT and MolFormer caches after the GPU preflight is approved.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/build_tahoe_encoder_caches.sh --gpu INDEX --ckpt DIR [options]

Options:
  --scgpt-python PATH  Python with scgpt 0.2.4 (default: SCGPT_PY or python3)
  --core-python PATH   CytoBridge core Python with transformers/RDKit (default: CORE_PY)

Set CYTOBRIDGE_LAUNCH_CONFIRMED=1 only after the recorded GPU preflight is approved.
EOF
}

GPU=""
CKPT=""
SCGPT_PY_ARG=""
CORE_PY_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --ckpt) CKPT="${2:-}"; shift 2 ;;
    --scgpt-python) SCGPT_PY_ARG="${2:-}"; shift 2 ;;
    --core-python) CORE_PY_ARG="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$GPU" || -z "$CKPT" ]]; then
  usage >&2
  exit 2
fi
if [[ "${CYTOBRIDGE_LAUNCH_CONFIRMED:-0}" != "1" ]]; then
  echo "cache launch refused: resource estimate has not been explicitly confirmed" >&2
  exit 3
fi

DEFAULT_PYTHON="$(command -v python3 || command -v python || true)"
SCGPT_PY="${SCGPT_PY_ARG:-${SCGPT_PY:-$DEFAULT_PYTHON}}"
CORE_PY="${CORE_PY_ARG:-${CORE_PY:-$SCGPT_PY}}"
for executable in "$SCGPT_PY" "$CORE_PY"; do
  [[ -x "$executable" ]] || { echo "Python is not executable: $executable" >&2; exit 2; }
done
SCGPT_PREFIX="$(cd "$(dirname "$(readlink -f "$SCGPT_PY")")/.." && pwd)"
CORE_PREFIX="$(cd "$(dirname "$(readlink -f "$CORE_PY")")/.." && pwd)"
for checkpoint_file in args.json vocab.json best_model.pt; do
  [[ -f "$CKPT/$checkpoint_file" ]] || {
    echo "Checkpoint file is missing: $CKPT/$checkpoint_file" >&2
    exit 2
  }
done

IFS=',' read -r UTIL MEMORY TOTAL < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
)
UTIL="${UTIL//[[:space:]]/}"
MEMORY="${MEMORY//[[:space:]]/}"
TOTAL="${TOTAL//[[:space:]]/}"
if (( UTIL >= 5 || MEMORY >= 1024 )); then
  echo "GPU $GPU is not eligible: utilization=${UTIL}%, memory=${MEMORY}/${TOTAL} MiB" >&2
  exit 4
fi

notify() {
  local level="$1"
  local message="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge Tahoe cache" \
      --text "$message" --level "$level" || true
  fi
}
trap 'notify error "Tahoe encoder cache generation failed on GPU $GPU"' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROTOCOL_DIR="data/processed/tahoe_accept"
H5AD="$PROTOCOL_DIR/tahoe_processed.h5ad"
SPLIT_DIR="$PROTOCOL_DIR/splits"
SMILES="$PROTOCOL_DIR/drug_smiles.csv"
CACHE_DIR="data/cache/tahoe_accept"
SCGPT_CACHE="$CACHE_DIR/scgpt_emb.npy"
SCGPT_PROVENANCE="${SCGPT_CACHE}.provenance.json"
SCGPT_VALIDATION="$CACHE_DIR/scgpt_cache_validation.json"
MOLFORMER_CACHE="$CACHE_DIR/molformer_emb.npz"
MOLFORMER_PROVENANCE="${MOLFORMER_CACHE}.provenance.json"

for input in "$H5AD" "$SPLIT_DIR" "$SMILES"; do
  [[ -e "$input" ]] || { echo "Missing Tahoe input: $input" >&2; exit 5; }
done
for output in \
  "$SCGPT_CACHE" "$SCGPT_PROVENANCE" "$SCGPT_VALIDATION" \
  "$MOLFORMER_CACHE" "$MOLFORMER_PROVENANCE"; do
  [[ ! -e "$output" ]] || {
    echo "Refusing to overwrite existing output: $output" >&2
    exit 5
  }
done
mkdir -p "$CACHE_DIR"

echo "[tahoe-cache] GPU $GPU eligible: utilization=${UTIL}%, memory=${MEMORY}/${TOTAL} MiB"
CUDA_VISIBLE_DEVICES="$GPU" \
LD_LIBRARY_PATH="$SCGPT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH=. "$SCGPT_PY" \
  cytobridge/encoders/scgpt_wrapper.py \
  --adata "$H5AD" \
  --out "$SCGPT_CACHE" \
  --ckpt "$CKPT" \
  --batch-size 32 \
  --device cuda \
  --precision fp32 \
  --seed 20260710
LD_LIBRARY_PATH="$CORE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$CORE_PY" scripts/validate_scgpt_cache.py \
  --h5ad "$H5AD" \
  --cache "$SCGPT_CACHE" \
  --provenance "$SCGPT_PROVENANCE" \
  --split-dir "$SPLIT_DIR" \
  --prefix tahoe \
  --out "$SCGPT_VALIDATION"
CUDA_VISIBLE_DEVICES="$GPU" \
LD_LIBRARY_PATH="$CORE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH=. "$CORE_PY" \
  -m cytobridge.encoders.molformer_wrapper \
  --smiles_csv "$SMILES" \
  --out "$MOLFORMER_CACHE" \
  --device cuda \
  --batch-size 64

trap - ERR
notify info "Tahoe scGPT and MolFormer caches passed"
echo "[tahoe-cache] PASS"
