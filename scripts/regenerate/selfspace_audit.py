"""Table 6: the reconstruction-aware audit.

Each predictor is scored in its own clean reconstruction of treated and
control pseudobulk. The well-posedness gate (Oracle = 1.0, Mean = 0.5) is
checked before any row is reported.
"""
from __future__ import annotations

import numpy as np

from eval.metrics import drug_discrimination_score, inter_drug_pearson, per_pair_spearman
from scripts.regenerate.constructions import cell_line_means
from scripts.regenerate.inputs import BundleLoader

TOP_K = 50
AUDIT_KEYS = {
    "cb": "t6.cb",
    "ridge": "t6.ridge",
    "chemcpa": "t6.chemcpa",
    "biolord": "t6.biolord",
    "biolord_ood": "t6.biolord_ood",
}


def _ddc(pred, true, cl) -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=TOP_K,
        metric="pearson",
    )


def audit_row(pred, true, cl) -> dict:
    score = _ddc(pred, true, cl)
    mean = cell_line_means(true, cl)
    mean_score = _ddc(mean, true, cl)
    oracle_score = _ddc(true, true, cl)
    return {
        "ondiag": float(score["on_diag_mean"]),
        "mean": float(per_pair_spearman(true, mean, top_k=TOP_K).mean()),
        "inter": float(inter_drug_pearson(pred, cl)),
        "auc": float(score["specificity_auc"]),
        "gap": float(score["gap"]),
        "oracle_auc": float(oracle_score["specificity_auc"]),
        "mean_auc": float(mean_score["specificity_auc"]),
        "well_posed": (
            float(oracle_score["specificity_auc"]) > 0.999
            and abs(float(mean_score["specificity_auc"]) - 0.5) < 0.06
        ),
    }


def audit_all(loader: BundleLoader, cb_perpair: dict | None, cb_pooled: dict | None) -> dict:
    out: dict = {}
    for key, prefix in AUDIT_KEYS.items():
        pred = loader.load_npy(f"baselines/{key}/pred.npy")
        true = loader.load_npy(f"baselines/{key}/true.npy")
        meta = loader.load_csv(f"baselines/{key}/meta.csv")
        if pred is None or true is None or meta is None:
            # CytoBridge row can fall back to the e6e7 per-pair matrices.
            if key == "cb" and cb_perpair is not None:
                pred, true, meta = (
                    cb_perpair["pred"],
                    cb_perpair["true"],
                    cb_perpair["meta"],
                )
            else:
                continue
        cl = meta["cell_line"].astype(str).to_numpy()
        row = audit_row(pred, true, cl)
        out[f"{prefix}.ondiag"] = row["ondiag"]
        out[f"{prefix}.mean"] = row["mean"]
        out[f"{prefix}.inter"] = row["inter"]
        out[f"{prefix}.auc"] = row["auc"]
        out[f"{prefix}.gap"] = row["gap"]
        out[f"{prefix}.oracle_auc"] = row["oracle_auc"]
        out[f"{prefix}.mean_auc"] = row["mean_auc"]
        out[f"{prefix}.well_posed"] = row["well_posed"]
    return out
