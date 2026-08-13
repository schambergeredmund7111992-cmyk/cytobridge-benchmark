"""Run validation-only Random/Oracle calibration before learned benchmark jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import conditional_rank_score
from eval.package_artifact import load_targets
from eval.reference_controls import build_reference_prediction

SEEDS = (11, 23, 42, 67, 101)


def calibrate_controls(protocol_dir: Path) -> dict:
    split_dir = protocol_dir / "splits"
    true, _ = load_targets(split_dir / "val_targets.npz")
    metadata = pd.read_csv(split_dir / "val_targets_metadata.csv")
    panels = {
        str(context): np.asarray(indices, dtype=int)
        for context, indices in json.loads(
            (split_dir / "training_gene_panels.json").read_text()
        ).items()
    }
    random_scores = []
    for seed in SEEDS:
        prediction = build_reference_prediction(
            "random", true, metadata, seed=seed
        )
        score = conditional_rank_score(
            prediction,
            true,
            metadata["context_id"],
            metadata["drug_id"],
            panels,
        ).summary["conditional_accuracy_drug_macro"]
        random_scores.append({"seed": seed, "conditional_accuracy": float(score)})
    oracle = build_reference_prediction("oracle", true, metadata, seed=0)
    oracle_score = conditional_rank_score(
        oracle,
        true,
        metadata["context_id"],
        metadata["drug_id"],
        panels,
    ).summary["conditional_accuracy_drug_macro"]
    random_mean = float(np.mean([row["conditional_accuracy"] for row in random_scores]))
    if not 0.40 <= random_mean <= 0.60:
        raise ValueError(f"Random validation calibration failed: {random_mean}.")
    if float(oracle_score) < 0.99:
        raise ValueError(f"Oracle validation calibration failed: {oracle_score}.")
    return {
        "status": "passed",
        "split": "validation",
        "random_seeds": list(SEEDS),
        "random_scores": random_scores,
        "random_mean": random_mean,
        "random_required_interval": [0.40, 0.60],
        "oracle": float(oracle_score),
        "oracle_required_minimum": 0.99,
        "test_targets_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite control calibration: {args.out}")
    result = calibrate_controls(args.protocol_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
