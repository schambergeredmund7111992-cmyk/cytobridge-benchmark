#!/usr/bin/env bash
# Build and validate both frozen sci-Plex scGPT caches. This performs inference only.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/build_sciplex_scgpt_caches.sh --gpu INDEX --ckpt DIR [options]

Options:
  --scgpt-python PATH  Python with scgpt 0.2.4 (default: SCGPT_PY or current python)
  --core-python PATH   Python with pandas/pyarrow/anndata (default: CORE_PY or scGPT Python)

Optional environment variables:
  SCGPT_PY  Python in the scGPT 0.2.4 environment
  CORE_PY   Python in the CytoBridge core environment
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
  exit 3
fi
echo "[scgpt-cache] GPU $GPU eligible: utilization=${UTIL}%, memory=${MEMORY}/${TOTAL} MiB"

notify() {
  local level="$1"
  local text="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge scGPT cache" --text "$text" --level "$level" || true
  fi
}
trap 'notify error "cache generation failed on GPU $GPU; inspect the detached log"' ERR

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
for protocol in drug_disjoint_v2 scaffold_disjoint_v2; do
  protocol_dir="data/processed/sciplex_accept/$protocol"
  h5ad="$protocol_dir/sciplex_processed.h5ad"
  split_dir="$protocol_dir/splits"
  cache_dir="data/cache/sciplex_accept/$protocol"
  cache="$cache_dir/scgpt_emb.npy"
  provenance="${cache}.provenance.json"
  validation="$cache_dir/scgpt_cache_validation.json"
  [[ -f "$h5ad" ]] || { echo "Missing processed h5ad: $h5ad" >&2; exit 4; }
  [[ -d "$split_dir" ]] || { echo "Missing split directory: $split_dir" >&2; exit 4; }
  mkdir -p "$cache_dir"
  for output in "$cache" "$provenance" "$validation"; do
    [[ ! -e "$output" ]] || {
      echo "Refusing to overwrite existing output: $output" >&2
      exit 4
    }
  done

  echo "[scgpt-cache] building $protocol"
  CUDA_VISIBLE_DEVICES="$GPU" \
  LD_LIBRARY_PATH="$SCGPT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH=. "$SCGPT_PY" \
    cytobridge/encoders/scgpt_wrapper.py \
    --adata "$h5ad" \
    --out "$cache" \
    --ckpt "$CKPT" \
    --batch-size 32 \
    --device cuda \
    --precision fp32 \
    --seed 20260710
  LD_LIBRARY_PATH="$CORE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$CORE_PY" scripts/validate_scgpt_cache.py \
    --h5ad "$h5ad" \
    --cache "$cache" \
    --provenance "$provenance" \
    --split-dir "$split_dir" \
    --out "$validation"
done

trap - ERR
notify info "both protocol-specific scGPT caches and validation reports passed"
echo "[scgpt-cache] PASS: both protocol caches validated"
