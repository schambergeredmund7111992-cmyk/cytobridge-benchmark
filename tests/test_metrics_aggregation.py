from __future__ import annotations

import numpy as np
from scipy import stats

from eval.metrics import per_pair_spearman
from eval.run_internal import aggregate_by_pair


def test_per_pair_spearman_topk_uses_aggregated_logfc_values():
    pred_mu = np.array([[200.0, 4.0, 8.0], [200.0, 4.0, 8.0]], dtype=np.float32)
    true_counts = np.array([[100.0, 10.0, 8.0], [100.0, 10.0, 8.0]], dtype=np.float32)
    ctrl_counts = np.array([[99.0, 0.0, 0.0], [99.0, 0.0, 0.0]], dtype=np.float32)

    preds, trues, _, _ = aggregate_by_pair(
        pred_mu=pred_mu,
        true_counts=true_counts,
        ctrl_counts=ctrl_counts,
        drug_ids=["drugA", "drugA"],
        cell_lines=["cellA", "cellA"],
    )

    top_idx = np.argsort(-np.abs(trues[0]))[:2]
    assert top_idx.tolist() == [1, 2]
    observed = per_pair_spearman(trues, preds, top_k=2)[0]
    expected = stats.spearmanr(trues[0, top_idx], preds[0, top_idx]).statistic
    assert np.isclose(observed, expected)
