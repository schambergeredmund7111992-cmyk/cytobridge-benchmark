"""Package row-aligned benchmark predictions as an immutable artifact."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from eval.artifacts import (
    SCHEMA_VERSION,
    sha256_file,
    sha256_gene_panel,
    sha256_json,
    write_artifact,
)


def load_targets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen true logFC matrix and ordered gene identifiers."""
    with np.load(path, allow_pickle=False) as payload:
        if "true" not in payload.files or "gene_ids" not in payload.files:
            raise ValueError(f"{path} must contain aligned true and gene_ids arrays.")
        true = np.asarray(payload["true"], dtype=float)
        gene_ids = np.asarray(payload["gene_ids"]).astype(str)
    if true.ndim != 2 or gene_ids.shape != (true.shape[1],):
        raise ValueError("Target matrix and ordered gene identifiers do not align.")
    return true, gene_ids


def load_artifact_metadata(path: Path, expected_rows: int) -> pd.DataFrame:
    """Select and order the five fields allowed by the artifact schema."""
    table = pd.read_csv(path)
    required = ["pair_id", "drug_id", "context_id", "split", "dataset"]
    if missing := set(required) - set(table.columns):
        raise ValueError(
            f"{path} is missing artifact metadata columns: {sorted(missing)}"
        )
    if len(table) != expected_rows:
        raise ValueError("Metadata rows do not align with frozen targets.")
    return table[required].copy()


def load_predictions(
    path: Path,
    *,
    expected_rows: int,
    expected_gene_ids: np.ndarray,
    expected_pair_ids: np.ndarray,
) -> np.ndarray:
    """Load predictions and validate optional alignment identifiers when present."""
    with np.load(path, allow_pickle=False) as payload:
        if "pred" not in payload.files:
            raise ValueError(f"{path} must contain a pred array.")
        pred = np.asarray(payload["pred"], dtype=float)
        if "gene_ids" in payload.files:
            observed_genes = np.asarray(payload["gene_ids"]).astype(str)
            if not np.array_equal(observed_genes, expected_gene_ids):
                raise ValueError(
                    "Prediction gene_ids differ from the frozen target order."
                )
        if "pair_ids" in payload.files:
            observed_pairs = np.asarray(payload["pair_ids"]).astype(str)
            if not np.array_equal(observed_pairs, expected_pair_ids.astype(str)):
                raise ValueError(
                    "Prediction pair_ids differ from the frozen target row order."
                )
    expected_shape = (expected_rows, len(expected_gene_ids))
    if pred.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {pred.shape} does not equal {expected_shape}."
        )
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    return pred


def build_provenance(
    *,
    dataset: str,
    split_name: str,
    model: str,
    seed: int,
    split_manifest: Path,
    config: Path,
    gene_panels: Path,
    gene_ids: np.ndarray,
    command: str,
    source_paths: Mapping[str, Path],
    checkpoint: Path | None = None,
    git_commit: str | None = None,
) -> dict:
    panel_payload = json.loads(gene_panels.read_text())
    normalized_panels = {
        str(context): [int(index) for index in indices]
        for context, indices in sorted(panel_payload.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split_name": split_name,
        "split_hash": sha256_file(split_manifest),
        "model": model,
        "seed": int(seed),
        "config_hash": sha256_file(config),
        "checkpoint_hash": sha256_file(checkpoint) if checkpoint is not None else None,
        "gene_panel_hash": sha256_gene_panel(gene_ids),
        "response_panel_hash": sha256_json(normalized_panels),
        "command": command,
        "git_commit": git_commit,
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
    }


def package_prediction_artifact(
    *,
    predictions: Path,
    targets: Path,
    metadata: Path,
    gene_panels: Path,
    split_manifest: Path,
    config: Path,
    output: Path,
    model: str,
    seed: int,
    checkpoint: Path | None = None,
    git_commit: str | None = None,
    command: str = "package_prediction_artifact",
    extra_sources: Mapping[str, Path] | None = None,
) -> None:
    true, gene_ids = load_targets(targets)
    table = load_artifact_metadata(metadata, len(true))
    datasets = table["dataset"].astype(str).unique().tolist()
    splits = table["split"].astype(str).unique().tolist()
    if len(datasets) != 1 or len(splits) != 1:
        raise ValueError(
            "A prediction artifact must contain exactly one dataset and split."
        )
    pred = load_predictions(
        predictions,
        expected_rows=len(true),
        expected_gene_ids=gene_ids,
        expected_pair_ids=table["pair_id"].to_numpy(),
    )
    source_paths = {
        "predictions": predictions,
        "targets": targets,
        "metadata": metadata,
        "gene_panels": gene_panels,
        "split_manifest": split_manifest,
        "config": config,
    }
    source_paths.update(extra_sources or {})
    provenance = build_provenance(
        dataset=datasets[0],
        split_name=splits[0],
        model=model,
        seed=seed,
        split_manifest=split_manifest,
        config=config,
        gene_panels=gene_panels,
        gene_ids=gene_ids,
        checkpoint=checkpoint,
        git_commit=git_commit,
        command=command,
        source_paths=source_paths,
    )
    write_artifact(
        output,
        pred=pred,
        true=true,
        gene_ids=gene_ids,
        metadata=table,
        provenance=provenance,
    )


def _source_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("Each --source must be a unique NAME=PATH value.")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--gene-panels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    command = " ".join(shlex.quote(part) for part in sys.argv)
    package_prediction_artifact(
        predictions=args.predictions,
        targets=args.targets,
        metadata=args.metadata,
        gene_panels=args.gene_panels,
        split_manifest=args.split_manifest,
        config=args.config,
        output=args.out,
        model=args.model,
        seed=args.seed,
        checkpoint=args.checkpoint,
        git_commit=args.git_commit,
        command=command,
        extra_sources=_source_arguments(args.source),
    )


if __name__ == "__main__":
    main()
