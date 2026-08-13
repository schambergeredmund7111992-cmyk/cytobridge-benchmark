#!/usr/bin/env bash
# Student-facing local gate. This verifies the lightweight CytoBridge path
# before any large data download or overnight GPU run.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

echo "[1/5] create tiny smoke data"
"$PYTHON" -m scripts.create_smoke_data --out data/smoke

echo "[2/5] run model forward/backward smoke"
"$PYTHON" -m scripts.run_model_smoke --data-dir data/smoke

echo "[3/5] run pytest"
"$PYTHON" -m pytest -q -rs

echo "[4/5] run ruff"
"$PYTHON" -m ruff check .

echo "[5/5] optional Lightning train smoke"
if "$PYTHON" -c "import hydra, pytorch_lightning" >/dev/null 2>&1; then
  "$PYTHON" train.py --config-name=train/smoke \
    trainer.max_epochs=1 \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    data.num_workers=0 \
    wandb.use=false
else
  echo "  - skipped: hydra-core and pytorch-lightning are not installed in this shell"
  echo "  - run after: conda env create -f env/environment.yml && conda activate cytobridge"
fi

echo "smoke test complete"
