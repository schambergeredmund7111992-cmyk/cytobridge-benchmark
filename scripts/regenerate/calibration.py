"""Fig. 4a: the injected-signal calibration curve on the pooled construction."""
from __future__ import annotations

import numpy as np

from eval.metrics import drug_discrimination_score
from scripts.regenerate.constructions import cell_line_means


def calibration_curve(true_pooled, cl, n_points: int = 11) -> dict:
    """Mix a fraction alpha of the true response into the cell-line Mean and
    score the mixture; returns alphas, aucs, and the effective alpha of the
    best audited configuration."""
    true = np.asarray(true_pooled, dtype=float)
    mean = cell_line_means(true, cl)
    alphas = np.linspace(0.0, 1.0, n_points)
    aucs = np.empty(n_points, dtype=float)
    for i, alpha in enumerate(alphas):
        mix = alpha * true + (1.0 - alpha) * mean
        aucs[i] = float(
            drug_discrimination_score(mix, true, cl, top_k=50, metric="pearson")[
                "specificity_auc"
            ]
        )
    return {"alphas": alphas, "aucs": aucs}


def effective_alpha(curve: dict, best_auc: float) -> float:
    aucs = np.asarray(curve["aucs"])
    alphas = np.asarray(curve["alphas"])
    if best_auc >= aucs[-1]:
        return float(alphas[-1])
    if best_auc <= aucs[0]:
        return float(alphas[0])
    return float(np.interp(best_auc, aucs, alphas))
