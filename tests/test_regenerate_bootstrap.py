"""Tests for the drug-clustered bootstrap, permutation null, and power analysis."""
from __future__ import annotations

import numpy as np

from scripts.regenerate.drug_clustered_auc_bootstrap import (
    between_drug_sd,
    delete_one_drug_sd,
    drug_clustered_auc_bootstrap,
    permutation_null,
    power_analysis,
)


def _synthetic(n_pairs=27, n_genes=300, seed=0, signal=0.1):
    rng = np.random.default_rng(seed)
    cl = np.array(["A549", "K562", "MCF7"] * (n_pairs // 3))
    drugs = np.array(
        [f"drug_{i // 3}" for i in range(n_pairs)]
    )
    base = rng.normal(size=(n_genes,))
    true = np.stack([base + rng.normal(0, 0.1, n_genes) for _ in range(n_pairs)])
    pred = signal * true + (1 - signal) * np.stack(
        [true[cl == cell].mean(axis=0) for cell in cl]
    )
    return pred, true, cl, drugs


def test_bootstrap_reproducible_with_seed():
    pred, true, cl, drugs = _synthetic()
    a = drug_clustered_auc_bootstrap(pred, true, cl, drugs, n_boot=50, seed=7)
    b = drug_clustered_auc_bootstrap(pred, true, cl, drugs, n_boot=50, seed=7)
    np.testing.assert_array_equal(a["draws"], b["draws"])
    assert a["ci_lo"] <= a["observed"] + 1e-9 or abs(a["ci_lo"] - a["observed"]) < 0.05
    assert 0.0 <= a["ci_lo"] <= a["ci_hi"] <= 1.0


def test_permutation_null_of_collapsed_predictor():
    pred, true, cl, drugs = _synthetic(signal=0.0)  # fully collapsed
    result = permutation_null(pred, true, cl, n_perm=100, seed=3)
    assert abs(result["observed"] - 0.5) < 1e-9
    assert abs(result["null_mean"] - 0.5) < 0.02
    assert result["p_value"] == 1.0  # nothing can beat a chance-level statistic


def test_delete_one_drug_sd_is_positive_and_bounded():
    pred, true, cl, drugs = _synthetic()
    value = delete_one_drug_sd(pred, true, cl, drugs)
    assert 0.0 <= value <= 1.0


def test_between_drug_sd_matches_hand_computation():
    pred, true, cl, drugs = _synthetic()
    from scripts.regenerate.drug_clustered_auc_bootstrap import per_anchor_scores
    anchors = per_anchor_scores(pred, true, cl)
    per_drug = []
    for drug in sorted(set(drugs.tolist())):
        rows = np.flatnonzero(drugs == drug)
        per_drug.append(float(anchors[rows].mean()))
    expected = float(np.std(per_drug, ddof=1))
    assert abs(between_drug_sd(pred, true, cl, drugs) - expected) < 1e-9


def test_power_analysis_reproduces_paper_numbers():
    """Sec 4.5: with a between-drug SD of 0.180, nine compounds give 91%
    power at AUC 0.70, 38% at 0.60, 13% at 0.55; 80% power needs 7, 26, 102."""
    result = power_analysis(0.180, n_drugs=9)
    assert 0.90 <= result["power70"] <= 0.92
    assert 0.36 <= result["power60"] <= 0.40
    assert 0.12 <= result["power55"] <= 0.14
    assert result["n80_at_70"] == 7
    assert result["n80_at_60"] == 26
    assert result["n80_at_55"] == 102
