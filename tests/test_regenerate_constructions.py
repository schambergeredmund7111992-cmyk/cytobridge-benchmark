"""Tests for the two vehicle constructions and their invariants."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.regenerate.constructions import (
    cell_line_means,
    check_invariants,
    conversion_delta,
    derive_pooled,
)


def _synthetic(n_pairs=27, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    true_pooled = rng.normal(size=(n_pairs, n_genes))
    true_perpair = true_pooled + rng.normal(0, 0.05, size=(n_pairs, n_genes))
    cl = np.array(["A549", "K562", "MCF7"] * (n_pairs // 3))
    pred_perpair = true_pooled * 0.1 + rng.normal(0, 0.5, size=(n_pairs, n_genes))
    return true_pooled, true_perpair, cl, pred_perpair


def test_delta_roundtrip_is_exact():
    true_pooled, true_perpair, cl, pred_perpair = _synthetic()
    delta = conversion_delta(true_perpair, true_pooled)
    pooled = derive_pooled(pred_perpair, delta)
    back = derive_pooled(pooled, -delta)
    np.testing.assert_allclose(back, pred_perpair, rtol=0, atol=1e-6)


def test_cell_line_means_are_constant_per_line():
    true_pooled, _, cl, _ = _synthetic()
    means = cell_line_means(true_pooled, cl)
    for cell in np.unique(cl):
        rows = means[cl == cell]
        assert rows.shape == (int(np.sum(cl == cell)), true_pooled.shape[1])
        assert bool((rows == rows[0]).all())


def test_no_drug_info_predictor_at_chance_in_pooled_space():
    true_pooled, true_perpair, cl, pred_perpair = _synthetic()
    pooled_preds = {
        "loss-only": derive_pooled(pred_perpair, conversion_delta(true_perpair, true_pooled))
    }
    problems = check_invariants(
        true_perpair=true_perpair,
        true_pooled=true_pooled,
        pooled_predictions=pooled_preds,
        cell_lines=cl,
    )
    assert problems == []


def test_invariant_catches_swapped_truth_labels():
    true_pooled, true_perpair, cl, pred_perpair = _synthetic()
    pooled_preds = {
        "loss-only": derive_pooled(
            pred_perpair, conversion_delta(true_perpair, true_pooled)
        )
    }
    # Swapping the two truth matrices (i.e. labelling the per-pair truth as
    # pooled) must be caught: the pooled no-drug-info predictor then leaves
    # chance and the per-pair one loses its inflation.
    problems = check_invariants(
        true_perpair=true_pooled,
        true_pooled=true_perpair,
        pooled_predictions=pooled_preds,
        cell_lines=cl,
    )
    assert problems != []


def test_delta_shape_mismatch_raises():
    with pytest.raises(ValueError):
        conversion_delta(np.zeros((27, 300)), np.zeros((26, 300)))
