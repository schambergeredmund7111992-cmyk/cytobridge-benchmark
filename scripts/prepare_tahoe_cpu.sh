#!/usr/bin/env bash
# Download the pinned Tahoe metadata, stream the bounded panel, and freeze train/val/test data.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_RUNNER="${CORE_RUNNER:-$ROOT/env/core_python.sh}"
if [[ ! -x "$CORE_RUNNER" ]]; then
  echo "Missing core Python runner: $CORE_RUNNER" >&2
  exit 2
fi

notify() {
  local level="$1"
  local message="$2"
  if [[ -x "$HOME/scripts/notify.py" ]]; then
    "$HOME/scripts/notify.py" --title "CytoBridge Tahoe CPU" \
      --text "$message" --level "$level" || true
  fi
}
trap 'notify error "Tahoe CPU preparation failed; inspect the detached log"' ERR

cd "$ROOT"
METADATA="data/raw/tahoe"
SELECTION="data/processed/tahoe_selection"
PANEL="data/raw/tahoe/selected_panel.h5ad"
PROTOCOL="data/processed/tahoe_accept"
SCIPLEX_GENES="data/processed/sciplex_accept/drug_disjoint_v2/gene_ids.txt"
GMT="data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt"

if [[ ! -f "$METADATA/metadata_provenance.json" ]]; then
  "$CORE_RUNNER" -m data.download_tahoe_metadata --out "$METADATA"
fi
if [[ ! -f "$SELECTION/selection.json" ]]; then
  "$CORE_RUNNER" -m data.select_tahoe_streaming \
    --obs-metadata "$METADATA/metadata/obs_metadata.parquet" \
    --sample-metadata "$METADATA/metadata/sample_metadata.parquet" \
    --drug-metadata "$METADATA/metadata/drug_metadata.parquet" \
    --out "$SELECTION"
fi
if [[ ! -f "$PANEL" ]]; then
  "$CORE_RUNNER" -m data.stream_tahoe_panel \
    --selected-cells "$SELECTION/selected_cells.parquet" \
    --gene-metadata "$METADATA/metadata/gene_metadata.parquet" \
    --dataset-revision 2dc5790 \
    --out "$PANEL"
fi
if [[ ! -d "$PROTOCOL" ]]; then
  "$CORE_RUNNER" -m data.preprocess_tahoe \
    --selected-h5ad "$PANEL" \
    --sciplex-gene-ids "$SCIPLEX_GENES" \
    --out "$PROTOCOL"
fi
if [[ ! -f "$PROTOCOL/pathway_names.txt" ]]; then
  "$CORE_RUNNER" -m data.pathway_gsea \
    --protocol-dir "$PROTOCOL" \
    --prefix tahoe \
    --gmt "$GMT"
fi
if [[ ! -f "$PROTOCOL/fit/fit_manifest.json" ]]; then
  "$CORE_RUNNER" -m data.combine_fit_splits \
    --protocol-dir "$PROTOCOL" \
    --prefix tahoe \
    --out "$PROTOCOL/fit"
fi

trap - ERR
notify info "Tahoe bounded panel, frozen splits, GSEA, and final-fit arrays passed"
echo "[tahoe-cpu] PASS"
