"""
eval/metrics.py
---------------
Evaluation metrics for CytoBridge.

All metrics expect:
    pred: [n_pairs, n_genes] predicted post-treatment logFC
    true: [n_pairs, n_genes] true post-treatment logFC

For per-pair metrics, we evaluate on top-K differentially expressed genes
ranked by |true logFC|, K=50 (matches scPerturBench convention).

Distributional metrics (E-distance) operate on cell-level distributions
rather than pseudobulk averages.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Per-pair correlation metrics
# ---------------------------------------------------------------------------
def per_pair_pearson(true: np.ndarray, pred: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Per-pair Pearson r on top-K DEGs."""
    r = []
    for t, p in zip(true, pred):
        idx = np.argsort(-np.abs(t))[:top_k]
        if np.std(p[idx]) < 1e-6 or np.std(t[idx]) < 1e-6:
            r.append(np.nan)
            continue
        r.append(stats.pearsonr(t[idx], p[idx]).statistic)
    return np.array(r)


def per_pair_spearman(true: np.ndarray, pred: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Per-pair Spearman rho on top-K DEGs."""
    r = []
    for t, p in zip(true, pred):
        idx = np.argsort(-np.abs(t))[:top_k]
        if np.std(p[idx]) < 1e-6 or np.std(t[idx]) < 1e-6:
            r.append(np.nan)
            continue
        r.append(stats.spearmanr(t[idx], p[idx]).statistic)
    return np.array(r)


# ---------------------------------------------------------------------------
# Week5: field-standard R² + drug-specific delta metric
# ---------------------------------------------------------------------------
def r2_per_pair(true: np.ndarray, pred: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Per-pair R² (coefficient of determination) on top-K DEGs by |true logFC|.
    This is the field-standard sci-Plex metric (chemCPA / Biolord / PerturbNet)."""
    out = []
    for t, p in zip(true, pred):
        idx = np.argsort(-np.abs(t))[:top_k]
        tt, pp = t[idx], p[idx]
        ss_tot = float(np.sum((tt - tt.mean()) ** 2))
        if ss_tot < 1e-12:
            out.append(np.nan)
            continue
        ss_res = float(np.sum((tt - pp) ** 2))
        out.append(1.0 - ss_res / ss_tot)
    return np.array(out)


def drug_specific_delta_spearman(
    true: np.ndarray, pred: np.ndarray, cell_lines, top_k: int = 50,
) -> np.ndarray:
    """Pre-registered STRINGENT metric: subtract the per-cell-line MEAN logFC
    from both pred and true, then per-pair Spearman on top-K DEGs of the
    residual. A model that only predicts the cell-line mean scores ~0 here,
    so this isolates genuine drug-specific signal (the linear-baseline-paradox
    bypass). Report ALONGSIDE the standard metric, never alone.
    """
    cl = np.asarray(cell_lines)
    true_r = np.asarray(true, dtype=float).copy()
    pred_r = np.asarray(pred, dtype=float).copy()
    for c in np.unique(cl):
        m = cl == c
        if m.sum() == 0:
            continue
        true_r[m] = true[m] - true[m].mean(axis=0, keepdims=True)
        pred_r[m] = pred[m] - pred[m].mean(axis=0, keepdims=True)
    # Per-pair Spearman on the residual's top-K DEGs. Convention (so a mean-only
    # predictor scores ~0, NOT NaN): if the TRUE residual is (near-)constant the
    # drug-specific signal is undefined -> NaN (dropped); if only the PRED residual
    # is constant, the model captured no drug-specific signal -> 0.0.
    out = []
    for t, p in zip(true_r, pred_r):
        idx = np.argsort(-np.abs(t))[:top_k]
        tt, pp = t[idx], p[idx]
        if np.std(tt) < 1e-6:
            out.append(np.nan)
        elif np.std(pp) < 1e-6:
            out.append(0.0)
        else:
            out.append(float(stats.spearmanr(tt, pp).statistic))
    return np.array(out)


# ---------------------------------------------------------------------------
# Week7 V1: off-diagonal drug-shuffle control — THE "is it real" test.
#   Applies to ANY per-pair vector output (expression logFC OR pathway gate).
# ---------------------------------------------------------------------------
def _corr_vec(a: np.ndarray, b: np.ndarray, metric: str = "pearson") -> float:
    """Pearson (or Spearman) between two 1-D vectors; NaN if either is constant."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if metric == "spearman":
        a = stats.rankdata(a)
        b = stats.rankdata(b)
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da < 1e-12 or db < 1e-12:
        return np.nan
    return float((a * b).sum() / (da * db))


def drug_discrimination_score(
    pred: np.ndarray,            # [n_pairs, D] predicted vectors (logFC or pathway)
    true: np.ndarray,            # [n_pairs, D] ground-truth vectors
    cell_lines,                  # [n_pairs] cell-line label per pair
    top_k: Optional[int] = 50,   # restrict to union of per-pair top-k |true| dims; None/>=D = all
    metric: str = "pearson",
) -> dict:
    """Off-diagonal drug-shuffle control: is a high per-pair corr DRUG-SPECIFIC,
    or just shared structure across all pairs (the Week5/Week6 collapse trap)?

    Within each cell line, build the cross-correlation  C[i, j] = corr(pred_i, true_j)
    over a FIXED dim set (so on- and off-diagonal are directly comparable), then:
        on_diag_mean    = mean_i  corr(pred_i, true_i)        (the usual per-pair r)
        off_diag_mean   = mean_{i!=j} corr(pred_i, true_j)    (the control)
        gap             = on_diag_mean - off_diag_mean        (drug-specific signal)
        specificity_auc = mean_i [ #{j!=i : C[i,i] > C[i,j]} / (m-1) ]

    A model whose output is (near-)constant across drugs — the diagnosed collapse,
    e.g. pathway_gate IDENTICAL for every drug, or inter-drug logFC Pearson 0.99 —
    has gap ~= 0 and specificity_auc ~= 0.5: pred_i matches OTHER drugs' truth just
    as well as its own. Genuine drug specificity has gap > 0 and auc -> 1.0.

    GATE: a per-pair r (faithfulness, prediction) is only citable as drug-specific
    if gap is clearly > 0 (wilcoxon_p_on_gt_off < 0.01) AND specificity_auc > ~0.7.
    Report this BESIDE every per-pair correlation, never the per-pair r alone.
    """
    cl = np.asarray(cell_lines)
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    D = pred.shape[1]
    use_all = (top_k is None) or (top_k >= D)
    on_all, off_all, auc_all = [], [], []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P, T = pred[m], true[m]
        if use_all:
            S = np.arange(D)
        else:
            sel = set()
            for i in range(m.size):
                sel.update(np.argsort(-np.abs(T[i]))[:top_k].tolist())
            S = np.array(sorted(sel), dtype=int)
        Ps, Ts = P[:, S], T[:, S]
        mm = m.size
        C = np.full((mm, mm), np.nan)
        for i in range(mm):
            for j in range(mm):
                C[i, j] = _corr_vec(Ps[i], Ts[j], metric)
        for i in range(mm):
            diag = C[i, i]
            offs = np.array([C[i, j] for j in range(mm) if j != i], dtype=float)
            offs = offs[~np.isnan(offs)]
            if np.isnan(diag) or offs.size == 0:
                continue
            on_all.append(float(diag))
            off_all.append(float(np.mean(offs)))
            auc_all.append(float(np.mean(diag > offs)))
    on_arr = np.array(on_all)
    off_arr = np.array(off_all)
    out = {
        "on_diag_mean": float(np.mean(on_arr)) if on_arr.size else float("nan"),
        "off_diag_mean": float(np.mean(off_arr)) if off_arr.size else float("nan"),
        "gap": float(np.mean(on_arr - off_arr)) if on_arr.size else float("nan"),
        "specificity_auc": float(np.mean(auc_all)) if auc_all else float("nan"),
        "n_pairs_scored": int(on_arr.size),
    }
    if on_arr.size >= 2 and np.any(on_arr != off_arr):
        try:
            out["wilcoxon_p_on_gt_off"] = float(
                stats.wilcoxon(on_arr, off_arr, alternative="greater").pvalue)
        except Exception:
            out["wilcoxon_p_on_gt_off"] = float("nan")
    else:
        out["wilcoxon_p_on_gt_off"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Week6 T6-3: drug-ranking metrics (the Option-E quantitative hook)
#   scores:    [n_queries, n_items] higher = predicted better
#   relevant:  [n_queries, n_items] bool ground-truth relevance (for Hit@K/MRR)
#   relevance: [n_queries, n_items] graded float relevance   (for NDCG)
# ---------------------------------------------------------------------------
def hit_at_k(scores: np.ndarray, relevant: np.ndarray, ks=(1, 3, 5)) -> dict:
    """Fraction of queries with >=1 relevant item in the top-k."""
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = np.take_along_axis(relevant.astype(bool), topk, axis=1).any(axis=1)
        out[int(k)] = float(hits.mean())
    return out


def mrr(scores: np.ndarray, relevant: np.ndarray) -> float:
    """Mean reciprocal rank of the first relevant item."""
    order = np.argsort(-scores, axis=1)
    rr = []
    rel = relevant.astype(bool)
    for q in range(scores.shape[0]):
        ranked = rel[q][order[q]]
        hits = np.flatnonzero(ranked)
        rr.append(1.0 / (hits[0] + 1) if hits.size else 0.0)
    return float(np.mean(rr))


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """Normalized DCG@k with graded relevance (0 .. perfect=1)."""
    k = min(k, scores.shape[1])
    order = np.argsort(-scores, axis=1)[:, :k]
    gains = np.take_along_axis(relevance.astype(float), order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum(axis=1)
    ideal = np.sort(relevance.astype(float), axis=1)[:, ::-1][:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    return float(np.mean(np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)))


# ---------------------------------------------------------------------------
# Week6 T6-1: collapse / scale diagnostics (print on EVERY eval artifact).
#   pred_logfc / true_logfc: [n_pairs, n_genes];  cell_lines: [n_pairs]
# ---------------------------------------------------------------------------
def inter_drug_pearson(pred_logfc: np.ndarray, cell_lines) -> float:
    """Mean pairwise Pearson of predicted logFC between different drugs WITHIN
    each cell line. The collapse meter: Week5 model = 0.97-0.99 while the true
    inter-drug Pearson is 0.13-0.42. The T6-1 gate is to push this < 0.7."""
    cl = np.asarray(cell_lines)
    vals = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P = pred_logfc[m]
        if np.std(P) < 1e-12:
            vals.append(1.0)               # exactly collapsed
            continue
        C = np.corrcoef(P)
        iu = np.triu_indices(m.size, 1)
        vals.append(float(np.nanmean(C[iu])))
    return float(np.mean(vals)) if vals else float("nan")


def scale_report(pred_logfc: np.ndarray, true_logfc: np.ndarray, cell_lines) -> dict:
    """pred/true std ratio per cell line (Week5 model = 4-11x mis-scaled) plus
    overall logFC mean/std/min/max for pred and true."""
    cl = np.asarray(cell_lines)
    per_cl = {}
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        ps, ts = float(pred_logfc[m].std()), float(true_logfc[m].std())
        per_cl[str(c)] = round(ps / ts, 4) if ts > 1e-12 else float("inf")

    def _stats(a):
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "pred_over_true_std_by_cell_line": per_cl,
        "pred_logfc": _stats(pred_logfc),
        "true_logfc": _stats(true_logfc),
        "inter_drug_pearson": inter_drug_pearson(pred_logfc, cell_lines),
    }


# ---------------------------------------------------------------------------
# Distributional metric: Energy distance
# ---------------------------------------------------------------------------
def e_distance(X: np.ndarray, Y: np.ndarray, max_n: int = 1000) -> float:
    """
    Energy distance between two empirical distributions X [n, d] and Y [m, d].
    Used in scPerturb / Tahoe-100M papers.
    Subsample to max_n cells per side for tractability.
    """
    rng = np.random.default_rng(42)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    dxx = cdist(X, X).mean()
    dyy = cdist(Y, Y).mean()
    dxy = cdist(X, Y).mean()
    return 2 * dxy - dxx - dyy


# ---------------------------------------------------------------------------
# Top-K retrieval accuracy
# ---------------------------------------------------------------------------
def top_k_retrieval(
    query_emb: np.ndarray,           # [n_queries, D]
    candidate_emb: np.ndarray,       # [n_cands, D]
    correct_idx: np.ndarray,         # [n_queries] index of correct candidate
    ks: list[int] = (1, 5, 10),
) -> dict[int, float]:
    """
    Given (control, treated) pairs as queries, retrieve correct drug from candidate pool.
    """
    sim = query_emb @ candidate_emb.T  # [n_q, n_c]
    ranks = (-sim).argsort(axis=1)
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        correct = (topk == correct_idx[:, None]).any(axis=1).mean()
        out[k] = float(correct)
    return out


# ---------------------------------------------------------------------------
# Pathway Precision@K (interpretability)
# ---------------------------------------------------------------------------
def pathway_precision_at_k(
    pred_pathways: np.ndarray,       # [n_pairs, K] predicted attribution
    true_pathways: np.ndarray,       # [n_pairs, K] GSEA pre-rank ground truth
    k: int = 5,
) -> float:
    """
    Mean over pairs of |top-k(pred) ∩ top-k(true)| / k.
    """
    out = []
    for p, t in zip(pred_pathways, true_pathways):
        pred_top = set(np.argsort(-p)[:k].tolist())
        true_top = set(np.argsort(-t)[:k].tolist())
        out.append(len(pred_top & true_top) / k)
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lower, upper) at (1-alpha) confidence level."""
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    boots = np.array([
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return statistic(values), float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))


# ---------------------------------------------------------------------------
# Paired Wilcoxon vs baseline
# ---------------------------------------------------------------------------
def paired_wilcoxon(
    metric_ours: np.ndarray,
    metric_baseline: np.ndarray,
    alternative: str = "greater",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    valid = (~np.isnan(metric_ours)) & (~np.isnan(metric_baseline))
    res = stats.wilcoxon(metric_ours[valid], metric_baseline[valid], alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size: P(a>b) - P(a<b)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (n_a * n_b))


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
def evaluation_report(
    true: np.ndarray,
    pred: np.ndarray,
    name: str = "method",
    baseline_per_pair_spearman: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """One-row summary report."""
    pearson = per_pair_pearson(true, pred)
    spearman = per_pair_spearman(true, pred)
    p_mean, p_lo, p_hi = bootstrap_ci(pearson)
    s_mean, s_lo, s_hi = bootstrap_ci(spearman)
    row = {
        "method": name,
        "pearson_mean": p_mean, "pearson_ci_lo": p_lo, "pearson_ci_hi": p_hi,
        "spearman_mean": s_mean, "spearman_ci_lo": s_lo, "spearman_ci_hi": s_hi,
        "n_pairs": int(np.sum(~np.isnan(spearman))),
    }
    if baseline_per_pair_spearman is not None:
        stat, pval = paired_wilcoxon(spearman, baseline_per_pair_spearman)
        row["wilcoxon_p_vs_baseline"] = pval
        row["cliffs_delta_vs_baseline"] = cliffs_delta(spearman, baseline_per_pair_spearman)
    return pd.DataFrame([row])
