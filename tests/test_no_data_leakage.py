"""
tests/test_no_data_leakage.py
-----------------------------
Critical sanity test: assert that train/val/test splits have NO overlapping drugs.

If this fails, the entire paper's results are invalid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


SPLITS_DIR = Path("data/processed/sciplex/splits")


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet generated")
def test_split_drug_disjoint():
    """Train, val, test must contain disjoint sets of drug IDs."""
    splits_meta = json.load(open(SPLITS_DIR / "internal_splits.json"))
    train_drugs = set(splits_meta["train_drugs"])
    val_drugs = set(splits_meta["val_drugs"])
    test_drugs = set(splits_meta["test_drugs"])
    assert train_drugs & val_drugs == set(), \
        f"Leakage train↔val: {train_drugs & val_drugs}"
    assert train_drugs & test_drugs == set(), \
        f"Leakage train↔test: {train_drugs & test_drugs}"
    assert val_drugs & test_drugs == set(), \
        f"Leakage val↔test: {val_drugs & test_drugs}"


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet generated")
def test_manifest_drug_consistency():
    splits_meta = json.load(open(SPLITS_DIR / "internal_splits.json"))
    for name, ref in [("train", "train_drugs"), ("val", "val_drugs"), ("test", "test_drugs")]:
        m = pd.read_parquet(SPLITS_DIR / f"sciplex_{name}.parquet")
        manifest_drugs = set(m["drug_id"].unique()) - {"DMSO"}
        ref_drugs = set(splits_meta[ref])
        assert manifest_drugs == ref_drugs, \
            f"{name} manifest drugs ({len(manifest_drugs)}) != ref ({len(ref_drugs)})"


@pytest.mark.skipif(not SPLITS_DIR.exists(), reason="splits not yet generated")
def test_no_treated_row_uses_self_as_control():
    """Regression: every non-control treated row must have control_cell_idx
    different from cell_idx. Pairing a treated cell with itself would feed
    the post-treatment embedding through the encoder as if it were the
    untreated baseline, leaking the answer into every prediction."""
    for name in ("train", "val", "test"):
        m = pd.read_parquet(SPLITS_DIR / f"sciplex_{name}.parquet")
        treated = m[m["drug_id"] != "DMSO"]
        leaks = (treated["cell_idx"] == treated["control_cell_idx"]).sum()
        assert leaks == 0, (
            f"{name}: {leaks} treated rows have control_cell_idx == cell_idx; "
            "these leak the treated state into the control input."
        )


def test_random_pair_baseline_near_zero():
    """A model that randomly pairs (cell, drug) should score Spearman ≈ 0.

    This is a positive control for our evaluation pipeline — if random pairs
    score significantly above 0, there's leakage somewhere.
    """
    import numpy as np
    from eval.metrics import per_pair_spearman
    rng = np.random.default_rng(0)
    n, g = 100, 500
    true = rng.normal(size=(n, g))
    pred = rng.normal(size=(n, g))   # totally random
    rho = per_pair_spearman(true, pred)
    assert abs(np.nanmean(rho)) < 0.05, \
        f"Random Spearman should be ≈0; got {np.nanmean(rho):.4f}. Check eval pipeline."


def test_internal_split_groups_duplicate_canonical_smiles_and_excludes_control():
    from data.benchmark_splits import make_drug_disjoint_v2

    compounds = pd.DataFrame(
        {
            "drug_id": ["drug_a1", "drug_a2", "drug_b", "drug_c"],
            "canonical_smiles": ["CCO", "CCO", "CCC", "CCN"],
            "scaffold_smiles": ["s1", "s1", "s2", "s3"],
            "eligible": [True, True, True, True],
        }
    )
    assignments = make_drug_disjoint_v2(compounds, seed=1).assignments
    splits = {
        name: set(assignments.loc[assignments["split"].eq(name), "drug_id"])
        for name in ("train", "val", "test")
    }

    split_sets = [splits[name] for name in ("train", "val", "test")]
    assert all("Vehicle" not in s for s in split_sets)
    assert sum("drug_a1" in s for s in split_sets) == 1
    assert sum("drug_a2" in s for s in split_sets) == 1
    assert any({"drug_a1", "drug_a2"} <= s for s in split_sets)
