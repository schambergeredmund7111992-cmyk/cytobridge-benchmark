#!/usr/bin/env bash
# Build immutable validation-only and final-refit AnnData exports for official baselines.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_PY="${CORE_PY:-$HOME/anaconda3/envs/cytobridge-accept-core/bin/python}"
EXTERNAL_PY="${CYTOBRIDGE_EXTERNAL_PYTHON:-$HOME/anaconda3/envs/cytobridge-accept-external/bin/python}"
for executable in "$CORE_PY" "$EXTERNAL_PY"; do
  [[ -x "$executable" ]] || { echo "Missing Python: $executable" >&2; exit 2; }
done
CORE_PREFIX="$(cd "$(dirname "$CORE_PY")/.." && pwd)"
EXTERNAL_PREFIX="$(cd "$(dirname "$EXTERNAL_PY")/.." && pwd)"

notify() {
  local level="$1"
  local message="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge external inputs" \
      --text "$message" --level "$level" || true
  fi
}
trap 'notify error "external input preparation failed"' ERR

cd "$ROOT"
build_export() {
  local protocol="$1"
  local mode="$2"
  local processed="data/processed/sciplex_accept/$protocol/sciplex_processed.h5ad"
  local splits="data/processed/sciplex_accept/$protocol/splits"
  local output="data/processed/external/$protocol/$mode.h5ad"
  local args=(
    -m data.export_external_benchmark
    --processed-h5ad "$processed"
    --split-dir "$splits"
    --out "$output"
    --with-rdkit2d
  )
  if [[ "$mode" == "selection" ]]; then
    args+=(--selection-only)
  else
    args+=(--final-refit)
  fi
  LD_LIBRARY_PATH="$CORE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$CORE_PY" "${args[@]}"
  LD_LIBRARY_PATH="$EXTERNAL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$EXTERNAL_PY" -m external.prepare_chemcpa_embeddings \
    --export "$output" \
    --out "data/processed/external/$protocol/${mode}_chemcpa_rdkit.parquet"
}

build_export drug_disjoint_v2 selection
build_export drug_disjoint_v2 final_refit
build_export scaffold_disjoint_v2 final_refit

trap - ERR
notify info "validation-only and final-refit external inputs passed"
echo "[external-inputs] PASS"
