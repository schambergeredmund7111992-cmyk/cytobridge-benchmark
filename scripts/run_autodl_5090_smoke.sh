#!/usr/bin/env bash
# End-to-end core smoke for sysu (Ampere) and AutoDL RTX 5090 (Blackwell).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CYTOBRIDGE_PYTHON:-python}"
OUT_DIR="${CYTOBRIDGE_SMOKE_OUT:-$ROOT/../experiments/gates/autodl-5090-smoke}"
REQUIRE_BLACKWELL="${CYTOBRIDGE_REQUIRE_BLACKWELL:-1}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

"$PYTHON" -m pip check

VERIFY_ARGS=(
  --expected-torch 2.7.1
  --required-cuda-prefix 12.8
  --out "$OUT_DIR/runtime.json"
)
if [[ "$REQUIRE_BLACKWELL" == "1" ]]; then
  VERIFY_ARGS+=(--require-blackwell)
else
  VERIFY_ARGS+=(--no-require-blackwell)
fi

"$PYTHON" -m scripts.verify_gpu_runtime "${VERIFY_ARGS[@]}"
"$PYTHON" -m scripts.create_smoke_data --out data/smoke
"$PYTHON" -m scripts.run_model_smoke --data-dir data/smoke --device cuda --precision fp32
"$PYTHON" -m scripts.run_model_smoke --data-dir data/smoke --device cuda --precision bf16
"$PYTHON" -m pytest -q -rs
"$PYTHON" -m ruff check .

"$PYTHON" train.py --config-name=train/smoke \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=32-true \
  data.num_workers=2 \
  data.multiprocessing_context=spawn \
  data.persistent_workers=true \
  data.prefetch_factor=1 \
  data.pin_memory=true \
  wandb.use=false \
  run_name=autodl_5090_smoke \
  ckpt.dirpath="$OUT_DIR/ckpts" \
  +run_metadata_path="$OUT_DIR/run_metadata.json" \
  2>&1 | tee "$OUT_DIR/lightning-smoke.log"

if ! grep -Eiq 'train/loss(_(step|epoch))?[^[:alnum:]]*[=:][[:space:]]*[+-]?[0-9]' \
  "$OUT_DIR/lightning-smoke.log"; then
  echo "GPU smoke did not emit a finite train/loss line." >&2
  exit 9
fi
if grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  "$OUT_DIR/lightning-smoke.log"; then
  echo "GPU smoke emitted NaN/Inf." >&2
  exit 10
fi

echo "[autodl-5090-smoke] PASS out=$OUT_DIR"
