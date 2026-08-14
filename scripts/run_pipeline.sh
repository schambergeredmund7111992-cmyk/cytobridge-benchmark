#!/usr/bin/env bash
# Local preparation gate only. Long training is intentionally not launched here.
set -euo pipefail

cd "$(dirname "$0")/.."
PHASE="${1:-smoke}"

case "$PHASE" in
  smoke)
    bash scripts/run_student_smoke.sh
    ;;
  sciplex)
    if [[ ! -d data/processed/sciplex_accept ]]; then
      python -m data.preprocess --config configs/data/sciplex.yaml
    else
      echo "[sciplex] immutable output exists; not rebuilding"
    fi
    for protocol in drug_disjoint_v2 scaffold_disjoint_v2; do
      target="data/processed/sciplex_accept/${protocol}/pathway_names.txt"
      if [[ ! -f "$target" ]]; then
        python -m data.pathway_gsea \
          --protocol-dir "data/processed/sciplex_accept/${protocol}" \
          --gmt data/raw/msigdb/h.all.v2024.1.Hs.symbols.gmt
      else
        echo "[sciplex] ${protocol} GSEA exists; not overwriting"
      fi
    done
    pytest -q tests/test_no_data_leakage.py
    ;;
  *)
    echo "usage: bash scripts/run_pipeline.sh [smoke|sciplex]" >&2
    exit 2
    ;;
esac

echo "[pipeline] preparation complete; no training was launched"
