#!/usr/bin/env bash
# Score one immutable prediction artifact under the frozen benchmark.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash scripts/launch_eval.sh ARTIFACT_DIR GENE_PANELS_JSON SCORED_DIR" >&2
  exit 2
fi

python -m eval.run_benchmark \
  --artifact "$1" \
  --gene-panels "$2" \
  --out "$3" \
  --n-boot 10000 \
  --n-permutations 10000 \
  --seed 7301
