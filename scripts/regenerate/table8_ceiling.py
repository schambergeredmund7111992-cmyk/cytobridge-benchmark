"""Table 8: the two reliability ceilings for the exact 27 scored conditions.

Technical ceiling: split each condition's cells into two random halves
(supp_T2 construction; needs the cell-level h5ad). Biological ceiling: the two
independent sci-Plex replicates, prepared on disjoint plates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from eval.metrics import drug_discrimination_score, per_pair_pearson

_SUPP_T2 = (
    Path(__file__).resolve().parents[2]
    / "manuscript" / "analysis" / "supp_T2"
)


def _ddc(pred, true, cl) -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=50,
        metric="pearson",
    )


def biological_ceiling(rep1, rep2, meta, drugs) -> dict:
    """rep1/rep2: [27,3000] pooled-vehicle logFC of the two replicates."""
    cl = meta["cell_line"].astype(str).to_numpy()
    drugs = np.asarray(drugs)
    d12 = _ddc(rep1, rep2, cl)
    d21 = _ddc(rep2, rep1, cl)
    combined = float(np.mean([d12["specificity_auc"], d21["specificity_auc"]]))
    gap = float(np.mean([d12["gap"], d21["gap"]]))
    ondiag_r = float(np.nanmean(per_pair_pearson(rep2, rep1, top_k=50)))
    # delete-one-drug jackknife on the combined AUC
    unique = sorted(set(drugs.tolist()), key=repr)
    loo = []
    for drug in unique:
        keep = drugs != drug
        loo.append(
            float(
                np.mean(
                    [
                        _ddc(rep1[keep], rep2[keep], cl[keep])["specificity_auc"],
                        _ddc(rep2[keep], rep1[keep], cl[keep])["specificity_auc"],
                    ]
                )
            )
        )
    loo = np.asarray(loo)
    n = len(unique)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return {
        "t8.biological.auc": combined,
        "t8.biological.se": se,
        "t8.biological.ci_lo": combined - 1.96 * se,
        "t8.biological.ci_hi": combined + 1.96 * se,
        "t8.biological.dir1": float(d12["specificity_auc"]),
        "t8.biological.dir2": float(d21["specificity_auc"]),
        "t8.biological.gap": gap,
        "t8.biological.ondiag_r": ondiag_r,
        "loo": [float(value) for value in loo],
    }


def technical_ceiling(h5ad_path, meta, *, n_splits: int = 20, seed0: int = 0) -> dict:
    """Split-half reliability on cell-level data (supp_T2 construction)."""
    import anndata

    if str(_SUPP_T2) not in sys.path:
        sys.path.insert(0, str(_SUPP_T2))
    from replicate_noise_ceiling import replicate_halves_from_cells

    adata = anndata.read_h5ad(h5ad_path)
    obs = adata.obs
    counts = np.asarray(
        adata.layers["counts"].toarray()
        if hasattr(adata.layers["counts"], "toarray")
        else adata.layers["counts"]
    )
    # The 27 scored conditions: the 9 held-out drugs at 10 uM / 24 h plus the
    # cell-line DMSO cells used as the shared vehicle.
    scored_drugs = set(meta["drug"].astype(str))
    drug_col = (
        "perturbation"
        if "perturbation" in obs.columns
        else ("drug" if "drug" in obs.columns else None)
    )
    if drug_col is None:
        raise ValueError("h5ad obs has no perturbation/drug column.")
    obs["_drug"] = obs[drug_col].astype(str)
    mask = (
        obs["_drug"].isin(scored_drugs)
        & obs["dose_value"].eq(10.0)
        & obs["time"].astype(str).eq("24")
    )
    all_cells_mask = mask | obs["_drug"].eq("DMSO")
    X = counts[all_cells_mask.to_numpy()].astype(np.float64)
    drugs = obs.loc[all_cells_mask, "_drug"].to_numpy().astype(str)
    cells = obs.loc[all_cells_mask, "cell_line"].to_numpy().astype(str)

    aucs, gaps = [], []
    for seed in range(seed0, seed0 + n_splits):
        pred_a, true_b, cl, _ = replicate_halves_from_cells(
            X, drugs, cells, "DMSO", seed=seed, min_cells=10
        )
        score = _ddc(pred_a, true_b, cl)
        aucs.append(float(score["specificity_auc"]))
        gaps.append(float(score["gap"]))
    aucs = np.asarray(aucs)
    gaps = np.asarray(gaps)
    return {
        "t8.technical.auc": float(aucs.mean()),
        "t8.technical.se": float(aucs.std(ddof=1)),
        "t8.technical.gap": float(gaps.mean()),
        "n_splits": int(len(aucs)),
    }
