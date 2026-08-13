"""Shared row-alignment and validation scoring for official external baselines."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from eval.validation import compute_validation_metrics


def _dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def paired_control_inputs(
    export: ad.AnnData,
    benchmark_split: str,
) -> tuple[ad.AnnData, np.ndarray]:
    """Return treated metadata with X replaced by each row's exact input control."""
    mask = (
        export.obs["benchmark_split"].astype(str).eq(benchmark_split)
        & export.obs["control"].astype(int).eq(0)
    )
    treated = export[mask].copy()
    if treated.n_obs == 0:
        raise ValueError(f"External export has no treated rows for {benchmark_split}.")
    lookup = {str(row_id): index for index, row_id in enumerate(export.obs_names.astype(str))}
    control_ids = treated.obs["input_control_row_id"].astype(str).to_numpy()
    missing = sorted(set(control_ids) - set(lookup))
    if missing:
        raise ValueError(f"Paired input controls are absent from export: {missing[:5]}")
    control_indices = np.asarray([lookup[row_id] for row_id in control_ids], dtype=int)
    control_x = export.X[control_indices]
    paired = ad.AnnData(
        X=control_x.copy(),
        obs=treated.obs.copy(),
        var=export.var.copy(),
    )
    paired.obs_names = treated.obs_names.copy()
    for key, values in treated.obsm.items():
        paired.obsm[key] = np.asarray(values).copy()
    return paired, control_indices


def validate_prediction_rows(
    predicted_log1p: np.ndarray,
    paired: ad.AnnData,
    manifest_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    expected_ids = manifest["treated_cell_id"].astype(str).to_numpy()
    observed_ids = paired.obs["source_cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError("External prediction rows differ from the frozen manifest order.")
    predicted_log1p = np.asarray(predicted_log1p)
    if predicted_log1p.shape != (len(manifest), paired.n_vars):
        raise ValueError(
            f"External prediction shape {predicted_log1p.shape} does not match "
            f"{(len(manifest), paired.n_vars)}."
        )
    if not np.isfinite(predicted_log1p).all():
        raise FloatingPointError("External model produced NaN or Inf predictions.")
    return manifest


def score_validation_predictions(
    predicted_log1p: np.ndarray,
    paired: ad.AnnData,
    manifest_path: Path,
    treated_counts_path: Path,
    truth_control_counts_path: Path,
    gene_panels_path: Path,
) -> dict[str, float]:
    manifest = validate_prediction_rows(predicted_log1p, paired, manifest_path)
    predicted_counts = np.expm1(np.maximum(np.asarray(predicted_log1p, dtype=float), 0.0))
    true_treated = np.load(treated_counts_path, mmap_mode="r")
    truth_control = np.load(truth_control_counts_path, mmap_mode="r")
    panels_payload = json.loads(gene_panels_path.read_text())
    panels = {
        str(context): np.asarray(indices, dtype=int)
        for context, indices in panels_payload.items()
    }
    return compute_validation_metrics(
        predicted_counts,
        true_treated,
        truth_control,
        manifest["drug_id"].astype(str).tolist(),
        manifest["context_id"].astype(str).tolist(),
        panels,
    )


def write_external_prediction_npz(
    path: Path,
    predicted_log1p: np.ndarray,
    paired: ad.AnnData,
    manifest_path: Path,
) -> dict:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite external predictions: {path}")
    validate_prediction_rows(predicted_log1p, paired, manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_log1p=np.asarray(predicted_log1p, dtype=np.float32),
        row_ids=paired.obs_names.astype(str).to_numpy(),
        gene_ids=paired.var_names.astype(str).to_numpy(),
    )
    return {
        "rows": int(paired.n_obs),
        "genes": int(paired.n_vars),
        "negative_fraction": float((np.asarray(predicted_log1p) < 0).mean()),
    }
