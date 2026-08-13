#!/bin/bash
# T2-2 Lambda Sweep RE-RUN — only failed combinations, with 15min timeout per job
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/CytoBridge/code
export LD_PRELOAD=/root/miniconda3/lib/libstdc++.so.6
cd /root/CytoBridge/code

TIMEOUT_MIN=15

echo "=== T2-2 Lambda Sweep RE-RUN (9 failed combinations, timeout=${TIMEOUT_MIN}min) ==="
echo ""

# Only the 9 failed combos
COMBOS=(
  "0.1 2.0"
  "0.5 0.1"
  "0.5 0.5"
  "0.5 1.0"
  "0.5 2.0"
  "1.0 0.1"
  "1.0 0.5"
  "1.0 1.0"
  "1.0 2.0"
)

TOTAL=${#COMBOS[@]}
N=0
FAILED_COMBOS=""

for combo in "${COMBOS[@]}"; do
  ld=$(echo $combo | awk '{print $1}')
  lc=$(echo $combo | awk '{print $2}')
  N=$((N + 1))
  CKPT_DIR="ckpts/t2_lambda_sweep/ld_${ld}_lc_${lc}"

  # Clean failed directory
  rm -rf "${CKPT_DIR}"

  echo "[${N}/${TOTAL}] lam_delta=${ld} lam_contrast=${lc} -> ${CKPT_DIR}"

  timeout ${TIMEOUT_MIN}m /root/miniconda3/bin/python train.py \
      model.residual_decoder=true \
      model.pool_mode=drug_query \
      loss.lam_direction=0.0 \
      loss.lam_delta=${ld} \
      loss.lam_contrast=${lc} \
      run_name=t2_lambda_sweep \
      wandb.use=false \
      trainer.max_epochs=1 \
      trainer.precision=32 \
      "ckpt.dirpath=${CKPT_DIR}" \
      2>&1 | tail -5

  RC=${PIPESTATUS[0]}
  if [ $RC -eq 124 ]; then
      echo "  TIMED OUT [${N}/${TOTAL}] — skipping"
      FAILED_COMBOS="${FAILED_COMBOS}  ld=${ld} lc=${lc} (timeout)\n"
  elif [ $RC -ne 0 ]; then
      echo "  FAILED (exit ${RC}) [${N}/${TOTAL}]"
      FAILED_COMBOS="${FAILED_COMBOS}  ld=${ld} lc=${lc} (exit ${RC})\n"
  else
      echo "  Done [${N}/${TOTAL}]"
  fi
  echo ""
done

echo ""
echo "=== Re-run complete ==="
if [ -n "$FAILED_COMBOS" ]; then
    echo ""
    echo "FAILED/TIMED OUT combinations:"
    echo -e "$FAILED_COMBOS"
fi
echo ""
echo "Checkpoints:"
for d in ckpts/t2_lambda_sweep/ld_*/; do
    best=$(ls "$d"epoch*val_spearman*.ckpt 2>/dev/null | head -1)
    if [ -n "$best" ]; then
        echo "  $(basename $d): $(basename $best)"
    else
        echo "  $(basename $d): NO CHECKPOINT"
    fi
done
