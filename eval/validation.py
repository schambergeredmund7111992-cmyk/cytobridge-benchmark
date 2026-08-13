"""Validation-only model-selection metrics shared by trainers and baseline runners."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.aggregation import aggregate_logfc_by_pair
from eval.metrics import conditional_rank_score, pair_own_spearman


def compute_validation_metrics(
    pred_treated: np.ndarray,
    true_treated: np.ndarray,
    truth_control: np.ndarray,
    drug_ids: list[str],
    context_ids: list[str],
    gene_panels: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute checkpoint-selection metrics on validation drugs only."""
    pred_logfc, true_logfc, metadata = aggregate_logfc_by_pair(
        pred_treated, true_treated, truth_control, drug_ids, context_ids
    )
    conditional = conditional_rank_score(
        pred_logfc,
        true_logfc,
        metadata["context_id"],
        metadata["drug_id"],
        gene_panels,
    )
    spearman = pair_own_spearman(true_logfc, pred_logfc, top_k=50)
    spearman_frame = pd.DataFrame(
        {"drug_id": metadata["drug_id"].astype(str).to_numpy(), "score": spearman}
    )
    return {
        "conditional_accuracy_drug_macro": float(
            conditional.summary["conditional_accuracy_drug_macro"]
        ),
        "pair_own_spearman_top50_drug_macro": float(
            spearman_frame.groupby("drug_id", sort=True)["score"].mean().mean()
        ),
    }
