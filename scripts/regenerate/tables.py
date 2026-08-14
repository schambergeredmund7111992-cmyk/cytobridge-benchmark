"""Table 3/4/5/7 computations from the pooled construction.

Every number is recomputed from the shipped matrices using the same scoring
functions the paper reports (eval.metrics.drug_discrimination_score and
eval.metrics.per_pair_spearman); no printed number is hard-coded here.
"""
from __future__ import annotations

import numpy as np

from eval.metrics import drug_discrimination_score, inter_drug_pearson, per_pair_spearman
from scripts.regenerate.constructions import cell_line_means
from scripts.regenerate.inputs import CONFIG_ORDER, BundleLoader

TOP_K = 50
GRID_KS = [20, 50, 100, None]
GRID_METRICS = ["pearson", "spearman"]


def _ddc(pred, true, cl, top_k=TOP_K, metric="pearson") -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=top_k,
        metric=metric,
    )


def table3(loader: BundleLoader) -> dict:
    counts = loader.load_csv("table3_cell_counts.csv")
    out: dict = {}
    if counts is None:
        return out
    for row in counts.itertuples(index=False):
        drug = str(row.drug).lower().replace(" ", "").replace("-", "")
        cell = str(row.cell_line).lower()
        out[f"t3.{drug}.{cell}"] = int(row.n_cells)
    return out


def table4(loader: BundleLoader, pooled_preds: dict, pooled_true: np.ndarray, cl) -> dict:
    out: dict = {}
    mean_train = loader.mean_predictor_train()
    if mean_train is not None and pooled_true is not None:
        out["t4.mean"] = float(per_pair_spearman(pooled_true, mean_train, top_k=TOP_K).mean())
    for name in CONFIG_ORDER:
        pred = pooled_preds.get(name)
        if pred is None:
            continue
        out[f"t4.cb_{_id(name)}"] = float(
            per_pair_spearman(pooled_true, pred, top_k=TOP_K).mean()
        )
    # External baselines are scored in their own clean reconstructions; their
    # self-space Spearman comes from the shipped scored metrics.
    for key, entry_id in (("ridge", "t4.ridge"), ("chemcpa", "t4.chemcpa")):
        value = _baseline_self_space(loader, key, "spearman")
        if value is not None:
            out[entry_id] = value
    return out


def table5(loader: BundleLoader, pooled_preds: dict, pooled_true: np.ndarray, cl,
           true_perpair: np.ndarray | None = None) -> dict:
    out: dict = {}
    for name in CONFIG_ORDER:
        pred = pooled_preds.get(name)
        if pred is None:
            continue
        score = _ddc(pred, pooled_true, cl)
        inter = float(inter_drug_pearson(pred, cl))
        base = _id(name)
        out[f"t5.{base}.inter"] = inter
        out[f"t5.{base}.auc"] = float(score["specificity_auc"])
        out[f"t5.{base}.gap"] = float(score["gap"])
        out[f"t5.{base}.p"] = float(score["wilcoxon_p_on_gt_off"])
    # No-drug-information predictor under both constructions.
    mean_pooled = cell_line_means(pooled_true, cl)
    pooled_score = _ddc(mean_pooled, pooled_true, cl)
    out["t5.no_drug_info.pooled.auc"] = float(pooled_score["specificity_auc"])
    if true_perpair is not None:
        from scripts.regenerate.constructions import (
            conversion_delta,
            no_drug_info_perpair,
        )

        delta = conversion_delta(true_perpair, pooled_true)
        perpair_predictor = no_drug_info_perpair(true_perpair, delta, cl)
        perpair_score = _ddc(perpair_predictor, true_perpair, cl)
        out["t5.no_drug_info.perpair.auc"] = float(perpair_score["specificity_auc"])
        out["t5.no_drug_info.perpair.gap"] = float(perpair_score["gap"])
        out["t5.no_drug_info.perpair.inter"] = float(
            inter_drug_pearson(perpair_predictor, cl)
        )
    # External baselines in their own spaces.
    for key, entry_id in (("chemcpa", "t5.chemcpa.auc"), ("biolord", "t5.biolord.auc")):
        value = _baseline_self_space(loader, key, "auc")
        if value is not None:
            out[entry_id] = value
    return out


def table7(loader: BundleLoader, pooled_preds: dict, pooled_true: np.ndarray, cl) -> dict:
    out: dict = {}
    model_aucs: list[float] = []
    for name in CONFIG_ORDER:
        pred = pooled_preds.get(name)
        if pred is None:
            continue
        for top_k in GRID_KS:
            for metric in GRID_METRICS:
                score = _ddc(pred, pooled_true, cl, top_k=top_k, metric=metric)
                model_aucs.append(float(score["specificity_auc"]))
    if model_aucs:
        out["t7.range_lo"] = float(np.min(model_aucs))
        out["t7.range_hi"] = float(np.max(model_aucs))
    return out


def _id(name: str) -> str:
    return {
        "loss-only": "loss_only",
        "drug-spec x1": "drugspec1",
        "drug-spec x3": "drugspec3",
        "drug-spec x5": "drugspec5",
        "norm-only": "norm_only",
        "low recon weight": "low_recon",
        "recovery baseline": "recovery_base",
    }[name]


def _baseline_self_space(loader: BundleLoader, key: str, quantity: str) -> float | None:
    """Read a self-space value from a shipped baseline scored run.

    Prefers recomputation from shipped matrices; falls back to the scored
    metrics.json recorded by the frozen evaluation pipeline.
    """
    pred = loader.load_npy(f"baselines/{key}/pred.npy")
    true = loader.load_npy(f"baselines/{key}/true.npy")
    meta = loader.load_csv(f"baselines/{key}/meta.csv")
    if pred is not None and true is not None and meta is not None:
        cl = meta["cell_line"].astype(str).to_numpy()
        if quantity == "spearman":
            return float(per_pair_spearman(true, pred, top_k=TOP_K).mean())
        return float(_ddc(pred, true, cl)["specificity_auc"])
    metrics = loader.baseline_metrics(key)
    if not metrics:
        return None
    if quantity == "spearman":
        for candidate in ("spearman50_ondiag", "spearman50", "rho50",
                          "mean_spearman50"):
            value = metrics.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
    else:
        for candidate in ("control_auc50", "specificity_auc", "auc50", "auc"):
            value = metrics.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
    return None
