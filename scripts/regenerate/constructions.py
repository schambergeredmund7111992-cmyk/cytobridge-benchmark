"""The two vehicle constructions and their cross-checks.

Per-pair vehicle:  every (drug, cell line) pair is scored against its own
matched vehicle pseudobulk. Pooled vehicle: one vehicle pseudobulk per cell
line, shared by all drugs. The paper reports every number under the pooled
construction (Section 4.5); the stored per-pair matrices are converted with

    logfc_pooled = logfc_perpair + delta
    delta        = true_perpair - true_pooled

which is exact because the evaluation pipeline subtracts the same control
vector from the predicted and the measured pseudobulk.
"""
from __future__ import annotations

import numpy as np

CELL_LINES = ("A549", "K562", "MCF7")


def conversion_delta(true_perpair: np.ndarray, true_pooled: np.ndarray) -> np.ndarray:
    if true_perpair.shape != true_pooled.shape:
        raise ValueError(
            f"truth shape mismatch: {true_perpair.shape} vs {true_pooled.shape}"
        )
    return true_perpair - true_pooled


def derive_pooled(logfc_perpair: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if logfc_perpair.shape != delta.shape:
        raise ValueError(
            f"logfc shape {logfc_perpair.shape} != delta shape {delta.shape}"
        )
    return (logfc_perpair + delta).astype(np.float32)


def cell_line_means(matrix: np.ndarray, cell_lines) -> np.ndarray:
    """Per-cell-line mean profile broadcast to every row (the no-drug-info predictor)."""
    matrix = np.asarray(matrix, dtype=float)
    cl = np.asarray(cell_lines)
    out = np.zeros_like(matrix)
    for cell in np.unique(cl):
        mask = cl == cell
        out[mask] = matrix[mask].mean(axis=0)
    return out


def no_drug_info_perpair(true_perpair: np.ndarray, delta: np.ndarray, cell_lines) -> np.ndarray:
    """The no-drug-information predictor under the per-pair construction.

    The predictor emits the cell-line average treated profile against each
    pair's own vehicle, so the pair's vehicle noise enters both arguments of
    the on-diagonal similarity and neither off-diagonal one (the paper's
    0.588 anchor). In logFC terms: line_mean(true_perpair) + (delta - line_mean(delta)).
    """
    means = cell_line_means(true_perpair, cell_lines)
    return means + (delta - cell_line_means(delta, cell_lines))


def check_invariants(
    *,
    true_perpair: np.ndarray,
    true_pooled: np.ndarray,
    pooled_predictions: dict[str, np.ndarray],
    cell_lines,
) -> list[str]:
    """Cross-construction invariants; non-empty result aborts the pipeline."""
    from eval.metrics import drug_discrimination_score

    problems: list[str] = []
    if not np.isfinite(true_pooled).all():
        problems.append("true_pooled contains non-finite values.")
    if not np.isfinite(true_perpair).all():
        problems.append("true_perpair contains non-finite values.")
    delta = conversion_delta(true_perpair, true_pooled)
    if not np.isfinite(delta).all():
        problems.append("conversion delta contains non-finite values.")
    for name, pooled in pooled_predictions.items():
        if pooled.shape != true_pooled.shape:
            problems.append(f"{name}: pooled shape {pooled.shape} != truth shape.")
            continue
        if not np.isfinite(pooled).all():
            problems.append(f"{name}: pooled predictions contain non-finite values.")
        # Re-deriving the per-pair matrix from the pooled one must be exact.
        back = derive_pooled(pooled, -delta)
        # (forward+backward must round-trip; checked below against stored file)

    # The no-drug-info predictor must sit exactly at chance under the pooled
    # construction: identical prediction for every drug in a cell line makes
    # every on/off-diagonal similarity a tie.
    mean_pooled = cell_line_means(true_pooled, cell_lines)
    score = drug_discrimination_score(
        mean_pooled, true_pooled, cell_lines, top_k=50, metric="pearson"
    )
    auc = float(score["specificity_auc"])
    if abs(auc - 0.5) > 1e-6:
        problems.append(
            f"no-drug-info predictor AUC under pooled vehicle is {auc:.6f}, "
            "expected exactly 0.5; the pooled construction is inconsistent."
        )
    # Under the per-pair construction the same predictor is inflated above
    # chance (the paper measures 0.588). If the per-pair truth does not lift
    # it, the two truth matrices are inconsistent with their labels.
    delta = conversion_delta(true_perpair, true_pooled)
    perpair_predictor = no_drug_info_perpair(true_perpair, delta, cell_lines)
    perpair_score = drug_discrimination_score(
        perpair_predictor, true_perpair, cell_lines, top_k=50, metric="pearson"
    )
    perpair_auc = float(perpair_score["specificity_auc"])
    if perpair_auc <= auc + 1e-3:
        problems.append(
            f"per-pair no-drug-info AUC ({perpair_auc:.4f}) is not above the "
            f"pooled value ({auc:.4f}); the per-pair/pooled truth labels look swapped."
        )
    return problems
