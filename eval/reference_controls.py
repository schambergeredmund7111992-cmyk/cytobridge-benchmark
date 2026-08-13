"""Generate Random, Oracle, and training-only Mean benchmark predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from eval.baselines.mean import fit_context_mean, predict_context_mean
from eval.package_artifact import load_targets


def consistent_drug_permutation(
    true: np.ndarray,
    metadata: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Permute drug labels once and apply the same mapping in every context."""
    required = {"drug_id", "context_id"}
    if missing := required - set(metadata.columns):
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    values = np.asarray(true, dtype=float)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError("Truth and metadata rows do not align.")
    table = metadata[["drug_id", "context_id"]].astype(str)
    if table.duplicated().any():
        raise ValueError("Random control requires one row per drug/context pair.")
    context_sets = [
        set(group["drug_id"]) for _, group in table.groupby("context_id", sort=True)
    ]
    if not context_sets or any(drugs != context_sets[0] for drugs in context_sets[1:]):
        raise ValueError("Random control requires the same drug set in every context.")
    drugs = sorted(context_sets[0])
    if len(drugs) < 2:
        raise ValueError("Random control requires at least two drugs.")
    shuffled = np.random.default_rng(seed).permutation(drugs).tolist()
    mapping = dict(zip(drugs, shuffled))
    lookup = {
        (row.context_id, row.drug_id): index
        for index, row in enumerate(table.itertuples(index=False))
    }
    indices = [
        lookup[(row.context_id, mapping[row.drug_id])]
        for row in table.itertuples(index=False)
    ]
    return values[np.asarray(indices, dtype=int)]


def build_reference_prediction(
    kind: str,
    eval_true: np.ndarray,
    eval_metadata: pd.DataFrame,
    *,
    seed: int,
    train_true: np.ndarray | None = None,
    train_context_ids: Sequence[object] | None = None,
) -> np.ndarray:
    normalized = kind.lower()
    if normalized == "oracle":
        return np.asarray(eval_true, dtype=float).copy()
    if normalized == "random":
        return consistent_drug_permutation(eval_true, eval_metadata, seed=seed)
    if normalized == "mean":
        if train_true is None or train_context_ids is None:
            raise ValueError(
                "Mean control requires training targets and context identifiers."
            )
        fitted = fit_context_mean(train_true, train_context_ids)
        return predict_context_mean(fitted, eval_metadata["context_id"])
    raise ValueError("kind must be one of: random, oracle, mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("random", "oracle", "mean"), required=True)
    parser.add_argument("--eval-targets", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path)
    parser.add_argument("--train-metadata", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    eval_true, gene_ids = load_targets(args.eval_targets)
    eval_metadata = pd.read_csv(args.eval_metadata)
    train_true = None
    train_contexts = None
    if args.kind == "mean":
        if args.train_targets is None or args.train_metadata is None:
            parser.error("--train-targets and --train-metadata are required for mean")
        train_true, train_gene_ids = load_targets(args.train_targets)
        if not np.array_equal(gene_ids, train_gene_ids):
            raise ValueError("Training and evaluation target gene orders differ.")
        train_contexts = pd.read_csv(args.train_metadata)["context_id"]
    pred = build_reference_prediction(
        args.kind,
        eval_true,
        eval_metadata,
        seed=args.seed,
        train_true=train_true,
        train_context_ids=train_contexts,
    )
    if args.out.exists():
        raise FileExistsError(
            f"Refusing to overwrite reference prediction file: {args.out}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pred=pred.astype(np.float32),
        pair_ids=np.asarray(eval_metadata["pair_id"].astype(str).tolist(), dtype=str),
        gene_ids=gene_ids,
    )
    print(json.dumps({"kind": args.kind, "seed": args.seed, "rows": len(pred)}))


if __name__ == "__main__":
    main()
