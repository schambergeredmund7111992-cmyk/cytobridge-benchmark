"""Leakage-safe context mean baseline.

The fitted state is a mapping from context to the mean *training target* log-fold-change.
Evaluation targets are accepted only by scoring code and never by :func:`fit_context_mean` or
:func:`predict_context_mean`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def fit_context_mean(
    train_true: np.ndarray,
    train_context_ids: Sequence[object],
) -> dict[str, np.ndarray]:
    """Fit one mean logFC vector per context from training targets only."""
    values = np.asarray(train_true, dtype=float)
    contexts = np.asarray([str(value) for value in train_context_ids], dtype=object)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("train_true must be a non-empty [pairs, genes] array.")
    if contexts.shape != (values.shape[0],):
        raise ValueError("train_context_ids must have one value per training row.")
    if not np.isfinite(values).all():
        raise ValueError("train_true must contain only finite values.")

    return {
        context: values[contexts == context].mean(axis=0)
        for context in sorted(set(contexts.tolist()))
    }


def predict_context_mean(
    fitted: Mapping[str, np.ndarray],
    eval_context_ids: Sequence[object],
) -> np.ndarray:
    """Repeat the frozen training mean for each requested evaluation context."""
    if not fitted:
        raise ValueError("fitted context means are empty.")
    rows = []
    n_genes = None
    for raw_context in eval_context_ids:
        context = str(raw_context)
        if context not in fitted:
            raise ValueError(
                f"Evaluation context {context!r} was absent from training data."
            )
        row = np.asarray(fitted[context], dtype=float)
        if row.ndim != 1 or not np.isfinite(row).all():
            raise ValueError(
                f"Training mean for context {context!r} is not a finite vector."
            )
        n_genes = row.size if n_genes is None else n_genes
        if row.size != n_genes:
            raise ValueError(
                "All fitted context means must have the same gene dimension."
            )
        rows.append(row)
    if not rows:
        return np.empty((0, int(n_genes or 0)), dtype=float)
    return np.stack(rows)


def save_context_mean(path: Path, fitted: Mapping[str, np.ndarray]) -> None:
    """Write a deterministic, human-auditable fitted baseline."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing fitted baseline: {path}")
    contexts = sorted(fitted)
    matrix = predict_context_mean(fitted, contexts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, context_ids=np.asarray(contexts), mean_logfc=matrix)


def load_context_mean(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"context_ids", "mean_logfc"}:
            raise ValueError(
                "Mean baseline NPZ must contain only context_ids and mean_logfc."
            )
        contexts = data["context_ids"].astype(str)
        matrix = np.asarray(data["mean_logfc"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(contexts):
        raise ValueError(
            "Mean baseline context_ids and mean_logfc shapes do not align."
        )
    return {context: matrix[i] for i, context in enumerate(contexts)}


def _read_target_inputs(
    targets_path: Path, metadata_path: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(targets_path, allow_pickle=False) as data:
        key = "true" if "true" in data.files else "targets"
        if key not in data.files:
            raise ValueError(
                f"{targets_path} must contain a 'true' or 'targets' array."
            )
        targets = np.asarray(data[key], dtype=float)
    metadata = pd.read_csv(metadata_path)
    if "context_id" not in metadata:
        raise ValueError(f"{metadata_path} must contain context_id.")
    if len(metadata) != len(targets):
        raise ValueError("Target rows and metadata rows do not align.")
    return targets, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a training-only context mean baseline."
    )
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    train_true, train_meta = _read_target_inputs(
        args.train_targets, args.train_metadata
    )
    fitted = fit_context_mean(train_true, train_meta["context_id"])
    save_context_mean(args.out, fitted)
    summary = {
        "n_training_pairs": int(len(train_true)),
        "n_contexts": int(len(fitted)),
        "n_genes": int(train_true.shape[1]),
        "uses_evaluation_truth": False,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
