"""Shared pseudobulk target construction for every evaluated model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def aggregate_logfc_by_pair(
    pred_treated: np.ndarray,
    true_treated: np.ndarray,
    truth_control: np.ndarray,
    drug_ids: Sequence[object],
    context_ids: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pseudobulk counts by drug/context, then apply one log1p difference.

    ``truth_control`` must come from the disjoint truth-reference vehicle pool. Model-input
    controls are intentionally absent from this API so they cannot leak into target construction.
    """
    # Preserve memory-mapped float32 arrays. Casting a full cell-by-gene matrix to
    # float64 here can create multi-gigabyte copies before pair-wise aggregation.
    pred = np.asarray(pred_treated)
    treated = np.asarray(true_treated)
    control = np.asarray(truth_control)
    if pred.ndim != 2 or treated.ndim != 2 or control.ndim != 2:
        raise ValueError(
            "pred_treated, true_treated, and truth_control must be 2-D arrays."
        )
    if pred.shape != treated.shape or pred.shape != control.shape:
        raise ValueError(
            f"Count arrays must share a shape; got {pred.shape}, {treated.shape}, {control.shape}."
        )
    if (
        not np.isfinite(pred).all()
        or not np.isfinite(treated).all()
        or not np.isfinite(control).all()
    ):
        raise ValueError("Count arrays must contain only finite values.")
    if (pred < 0).any() or (treated < 0).any() or (control < 0).any():
        raise ValueError("Count arrays must be non-negative before log1p aggregation.")
    drugs = np.asarray([str(value) for value in drug_ids], dtype=object)
    contexts = np.asarray([str(value) for value in context_ids], dtype=object)
    if drugs.shape != (len(pred),) or contexts.shape != (len(pred),):
        raise ValueError("drug_ids and context_ids must have one value per count row.")

    row_metadata = pd.DataFrame({"drug_id": drugs, "context_id": contexts})
    pair_pred = []
    pair_true = []
    pair_records = []
    for pair_number, ((drug, context), index) in enumerate(
        row_metadata.groupby(["drug_id", "context_id"], sort=True).groups.items()
    ):
        take = np.asarray(list(index), dtype=int)
        pred_pb = pred[take].mean(axis=0)
        treated_pb = treated[take].mean(axis=0)
        control_pb = control[take].mean(axis=0)
        pair_pred.append(np.log1p(pred_pb) - np.log1p(control_pb))
        pair_true.append(np.log1p(treated_pb) - np.log1p(control_pb))
        pair_records.append(
            {
                "pair_id": f"{drug}::{context}",
                "drug_id": drug,
                "context_id": context,
                "n_cells": int(len(take)),
                "pair_index": pair_number,
            }
        )
    if not pair_pred:
        raise ValueError("No drug/context pairs were available for aggregation.")
    return np.stack(pair_pred), np.stack(pair_true), pd.DataFrame(pair_records)
