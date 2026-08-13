#!/bin/bash
# T2-2: Lambda Sweep — run each combination separately to avoid Hydra multirun encoding issues
set -e
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/CytoBridge/code
export LD_PRELOAD=/root/miniconda3/lib/libstdc++.so.6
cd /root/CytoBridge/code

echo "=== T2-2 Lambda Sweep (sequential) ==="
echo ""

LAM_DELTA_VALS=(0.0 0.1 0.5 1.0)
LAM_CONTRAST_VALS=(0.1 0.5 1.0 2.0)
TOTAL=$((${#LAM_DELTA_VALS[@]} * ${#LAM_CONTRAST_VALS[@]}))
N=0

for ld in "${LAM_DELTA_VALS[@]}"; do
  for lc in "${LAM_CONTRAST_VALS[@]}"; do
    N=$((N + 1))
    CKPT_DIR="ckpts/t2_lambda_sweep/ld_${ld}_lc_${lc}"
    echo "[${N}/${TOTAL}] lam_delta=${ld} lam_contrast=${lc} -> ${CKPT_DIR}"

    /root/miniconda3/bin/python train.py \
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

    echo "  Done [${N}/${TOTAL}]"
    echo ""
  done
done

echo ""
echo "=== Sweep complete ==="
echo "Results directories:"
ls -d ckpts/t2_lambda_sweep/ld_*/ 2>/dev/null
echo ""
echo "Checkpoints per combination:"
for d in ckpts/t2_lambda_sweep/ld_*/; do
    best=$(ls "$d"epoch*val_spearman*.ckpt 2>/dev/null | head -1)
    if [ -n "$best" ]; then
        echo "  $(basename $d): $(basename $best)"
    else
        echo "  $(basename $d): NO CHECKPOINT"
    fi
done
