"""
tests/test_no_data_leakage.py
-----------------------------
Critical sanity test: assert that train/val/test splits have NO overlapping drugs.

If this fails, the entire paper's results are invalid.

The assertions below target the frozen pipeline output under
data/processed/sciplex_accept/{protocol}/ (built by
`bash scripts/run_pipeline.sh sciplex`). If that output is missing the tests
FAIL rather than skip: a green run with no pipeline data would be a false green.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ACCEPT_ROOT = Path("data/processed/sciplex_accept")
SPLIT_NAMES = ("train", "val", "test")


def _protocol_dirs() -> list[Path]:
    if not ACCEPT_ROOT.exists():
        return []
    return sorted(
        path
        for path in ACCEPT_ROOT.iterdir()
        if path.is_dir() and (path / "split_manifest.json").is_file()
    )


def _require_protocol_dirs() -> list[Path]:
    dirs = _protocol_dirs()
    if not dirs:
        raise AssertionError(
            "No frozen sciplex_accept protocol output found under "
            f"{ACCEPT_ROOT}. Build it with `bash scripts/run_pipeline.sh sciplex` "
            "first. This leakage gate must never be skipped."
        )
    return dirs


def _assert_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{context} is missing column(s): {missing}")


def test_split_drug_disjoint():
    """Train, val, test must contain disjoint sets of canonical compounds."""
    for proto_dir in _require_protocol_dirs():
        assignments = pd.read_csv(proto_dir / "split_assignments.csv")
        _assert_columns(
            assignments,
            {"drug_id", "canonical_smiles", "split"},
            f"{proto_dir.name}/split_assignments.csv",
        )
        seen: dict[str, str] = {}
        for drug_id, smiles, split in assignments[
            ["drug_id", "canonical_smiles", "split"]
        ].itertuples(index=False):
            smiles = str(smiles)
            if smiles in seen and seen[smiles] != split:
                raise AssertionError(
                    f"{proto_dir.name}: canonical compound {smiles!r} appears in "
                    f"both {seen[smiles]} ({drug_id}) and {split}; drug-disjoint "
                    "split violated."
                )
            seen[smiles] = str(split)
        for split in SPLIT_NAMES:
            if not (assignments["split"].astype(str) == split).any():
                raise AssertionError(f"{proto_dir.name}: split {split!r} is empty.")


def test_manifest_drug_consistency():
    """Each split parquet must contain exactly the drugs assigned to that split."""
    for proto_dir in _require_protocol_dirs():
        assignments = pd.read_csv(proto_dir / "split_assignments.csv")
        splits_dir = proto_dir / "splits"
        for split in SPLIT_NAMES:
            ref = set(
                assignments.loc[
                    assignments["split"].astype(str).eq(split), "drug_id"
                ].astype(str)
            )
            manifest = pd.read_parquet(splits_dir / f"sciplex_{split}.parquet")
            _assert_columns(
                manifest,
                {"drug_id", "cell_idx", "control_cell_idx"},
                f"{proto_dir.name}/sciplex_{split}.parquet",
            )
            manifest_drugs = set(manifest["drug_id"].astype(str))
            assert manifest_drugs == ref, (
                f"{proto_dir.name}/{split}: manifest drugs ({len(manifest_drugs)}) "
                f"!= assignment ({len(ref)})"
            )
            assert "DMSO" not in manifest_drugs, (
                f"{proto_dir.name}/{split}: manifest contains control rows."
            )


def test_no_treated_row_uses_self_as_control():
    """Regression: every treated row must have control_cell_idx different from
    cell_idx. Pairing a treated cell with itself would feed the post-treatment
    embedding through the encoder as if it were the untreated baseline, leaking
    the answer into every prediction."""
    for proto_dir in _require_protocol_dirs():
        splits_dir = proto_dir / "splits"
        for split in SPLIT_NAMES:
            manifest = pd.read_parquet(splits_dir / f"sciplex_{split}.parquet")
            leaks = (manifest["cell_idx"] == manifest["control_cell_idx"]).sum()
            assert int(leaks) == 0, (
                f"{proto_dir.name}/{split}: {leaks} rows have "
                "control_cell_idx == cell_idx; these leak the treated state "
                "into the control input."
            )
            same_id = (
                manifest["treated_cell_id"].astype(str)
                == manifest["input_control_cell_id"].astype(str)
            ).sum()
            assert int(same_id) == 0, (
                f"{proto_dir.name}/{split}: {same_id} rows pair a treated cell "
                "with itself as input control."
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
