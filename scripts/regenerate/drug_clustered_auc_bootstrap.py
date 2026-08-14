"""Section 4.5 uncertainty quantities (Fig. 4e and the permutation null).

Drug-clustered bootstrap: resample the nine held-out compounds with
replacement, keeping each drug's three cell lines together, and recompute the
discrimination AUC on every resampled set (the negative pool is rebuilt with
the sample). Permutation null: shuffle the predicted rows within each cell
line, destroying the drug pairing while preserving cell-line structure.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from eval.metrics import drug_discrimination_score

TOP_K = 50
METRIC = "pearson"


def _auc(pred, true, cl) -> float:
    return float(
        drug_discrimination_score(
            np.asarray(pred, dtype=float),
            np.asarray(true, dtype=float),
            np.asarray(cl),
            top_k=TOP_K,
            metric=METRIC,
        )["specificity_auc"]
    )


def _drug_rows(drugs: np.ndarray, sampled: np.ndarray) -> list[int]:
    rows: list[int] = []
    for drug in sampled:
        rows.extend(int(i) for i in np.flatnonzero(drugs == drug))
    return rows


def drug_clustered_auc_bootstrap(
    pred, true, cl, drugs, *, n_boot: int = 1000, seed: int = 7301
) -> dict:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    unique = np.asarray(sorted(set(drugs.tolist()), key=repr))
    rng = np.random.default_rng(seed)
    observed = _auc(pred, true, cl)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = _drug_rows(drugs, sampled)
        draws[i] = _auc(pred[rows], true[rows], cl[rows])
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    return {
        "observed": observed,
        "ci_lo": lo,
        "ci_hi": hi,
        "bootstrap_mean": float(draws.mean()),
        "draws": draws,
        "seed": seed,
        "n_boot": n_boot,
    }


def permutation_null(
    pred, true, cl, *, n_perm: int = 1000, seed: int = 7301
) -> dict:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    rng = np.random.default_rng(seed)
    observed = _auc(pred, true, cl)
    null = np.empty(n_perm, dtype=float)
    shuffled = pred.copy()
    for i in range(n_perm):
        for cell in np.unique(cl):
            mask = cl == cell
            shuffled[mask] = rng.permutation(shuffled[mask])
        null[i] = _auc(shuffled, true, cl)
    p_value = float((int(np.sum(null >= observed)) + 1) / (n_perm + 1))
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_value": p_value,
        "null": null,
        "seed": seed,
        "n_perm": n_perm,
    }


def delete_one_drug_sd(pred, true, cl, drugs) -> float:
    """Standard deviation of the AUC across delete-one-drug subsets."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    unique = sorted(set(drugs.tolist()), key=repr)
    values = []
    for drug in unique:
        keep = drugs != drug
        values.append(_auc(pred[keep], true[keep], cl[keep]))
    return float(np.std(values, ddof=1))


def per_anchor_scores(pred, true, cl, top_k=TOP_K, metric=METRIC) -> np.ndarray:
    """Per-anchor discrimination scores (fraction of off-diagonal similarities
    the anchor beats within its cell line)."""
    from eval.metrics import _corr_vec

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    out = np.full(len(pred), np.nan, dtype=float)
    for cell in np.unique(cl):
        rows = np.flatnonzero(cl == cell)
        if rows.size < 2:
            continue
        P, T = pred[rows], true[rows]
        if top_k is None or top_k >= P.shape[1]:
            panel = np.arange(P.shape[1])
        else:
            panel = sorted(
                set().union(
                    *[set(np.argsort(-np.abs(T[i]))[:top_k].tolist()) for i in range(len(rows))]
                )
            )
        Ps, Ts = P[:, panel], T[:, panel]
        C = np.array(
            [[_corr_vec(Ps[i], Ts[j], metric) for j in range(len(rows))]
             for i in range(len(rows))]
        )
        for position, i in enumerate(rows):
            offs = np.delete(C[position], position)
            offs = offs[np.isfinite(offs)]
            if offs.size:
                out[i] = float(np.mean(C[position, position] > offs))
    return out


def between_drug_sd(pred, true, cl, drugs) -> float:
    """Standard deviation across drugs of the per-drug mean anchor score
    (three cell lines per drug), as in Sec 4.5."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    anchors = per_anchor_scores(pred, true, cl)
    per_drug = []
    for drug in sorted(set(drugs.tolist()), key=repr):
        rows = np.flatnonzero(drugs == drug)
        per_drug.append(float(anchors[rows].mean()))
    return float(np.std(per_drug, ddof=1))


def power_analysis(between_drug_sd: float, n_drugs: int = 9) -> dict:
    """Power against the null mean 0.5 with the between-drug SD, using the
    normal approximation (Phi(effect / (sd / sqrt(n)) - 1.96)), the
    calculation reported in Sec 4.5."""

    def power(effect: float, n: int) -> float:
        z = effect / (between_drug_sd / np.sqrt(n)) - 1.96
        return float(stats.norm.cdf(z))

    def n_for_power(effect: float, target: float = 0.80) -> int:
        z80 = float(stats.norm.ppf(target))
        required = ((z80 + 1.96) * between_drug_sd / effect) ** 2
        return int(np.ceil(required))

    return {
        "power70": power(0.20, n_drugs),
        "power60": power(0.10, n_drugs),
        "power55": power(0.05, n_drugs),
        "n80_at_70": n_for_power(0.20),
        "n80_at_60": n_for_power(0.10),
        "n80_at_55": n_for_power(0.05),
    }
