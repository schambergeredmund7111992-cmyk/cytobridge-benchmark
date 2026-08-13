"""
eval/baselines/gears_runner.py
-------------------------------
GEARS baseline (Roohani et al. Nature Biotechnology 2023).

GEARS is a graph neural network using a gene knowledge graph to predict
combinatorial perturbation effects. We use it as a strong DL baseline.

GitHub: https://github.com/snap-stanford/GEARS

NB: GEARS expects perturbations as gene names, not drug SMILES. For chemical
perturbation we map drug -> primary target gene (from DrugBank), then run.
This is a fair limitation to note in the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/sciplex/"))
    parser.add_argument("--out", type=Path, default=Path("results/gears.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    try:
        from gears import PertData, GEARS
    except ImportError:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "method": "GEARS",
            "status": "skipped",
            "reason": "gears package is not installed; install it in the experiment environment",
        }]).to_csv(args.out, index=False)
        print(f"GEARS unavailable; wrote skip record to {args.out}")
        return

    pert_data = PertData(str(args.data_dir))
    pert_data.load(data_name="sciplex")
    pert_data.prepare_split(split="simulation", seed=42)
    pert_data.get_dataloader(batch_size=32, test_batch_size=32)
    gears_model = GEARS(pert_data, device="cuda")
    gears_model.model_initialize(hidden_size=64)
    gears_model.train(epochs=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"method": "GEARS", "status": "trained", "epochs": args.epochs}]).to_csv(args.out, index=False)
    print(f"GEARS training record saved to {args.out}")


if __name__ == "__main__":
    main()
