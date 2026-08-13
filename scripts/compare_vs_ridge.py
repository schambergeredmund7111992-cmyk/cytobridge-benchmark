#!/usr/bin/env python
"""
compare_vs_ridge.py
-------------------
Rigorous per-pair comparison of one CytoBridge run vs the ridge baseline,
on the SAME 27 (drug, cell_line) pairs, in logFC-delta space.

This is the Stage-0 reconciliation artefact: it produces the numbers that were
MISSING from the package — paired Wilcoxon p and a bootstrap CI on the
*paired* Δ — so that any "v1/v2 vs ridge" headline is reproducible from a
single command.

Reuses the canonical metric utilities in eval/metrics.py (bootstrap_ci,
paired_wilcoxon, cliffs_delta) so the statistics match the rest of the repo.

Run from the repo `code/` directory so `from eval.metrics import ...` resolves:

    cd code
    python /path/to/compare_vs_ridge.py \
        --model_csv results/cytobridge_v1_full_ridge.csv \
        --ridge_csv results/ridge_baseline.csv \
        --label v1_full

Both CSVs must have columns: drug, cell_line, spearman_top50, pearson_top50.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from eval.metrics import bootstrap_ci, paired_wilcoxon, cliffs_delta
except Exception:  # pragma: no cover - fallback if not run from code/
    from scipy import stats

    def bootstrap_ci(values, statistic=np.mean, n_boot=1000, alpha=0.05, seed=42):
        rng = np.random.default_rng(seed)
        v = values[~np.isnan(values)]
        boots = np.array([statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)])
        return statistic(v), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))

    def paired_wilcoxon(a, b, alternative="greater"):
        valid = (~np.isnan(a)) & (~np.isnan(b))
        res = stats.wilcoxon(a[valid], b[valid], alternative=alternative)
        return float(res.statistic), float(res.pvalue)

    def cliffs_delta(a, b):
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        gt = (a[:, None] > b[None, :]).sum()
        lt = (a[:, None] < b[None, :]).sum()
        return float((gt - lt) / (len(a) * len(b)))


def bootstrap_paired_delta(a, b, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap CI on the PAIRED mean difference mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    valid = (~np.isnan(a)) & (~np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = a[idx].mean() - b[idx].mean()
    return float(a.mean() - b.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_csv", required=True, type=Path)
    ap.add_argument("--ridge_csv", required=True, type=Path)
    ap.add_argument("--label", default="model")
    ap.add_argument("--metric", default="spearman_top50",
                    choices=["spearman_top50", "pearson_top50"])
    ap.add_argument("--alpha_bonferroni", type=float, default=0.01,
                    help="Project significance threshold (0.05/5 baselines).")
    args = ap.parse_args()

    m = pd.read_csv(args.model_csv)
    r = pd.read_csv(args.ridge_csv)
    # ridge_baseline.csv may use 'drug' or 'drug_id'; normalize.
    for df in (m, r):
        if "drug" not in df.columns and "drug_id" in df.columns:
            df.rename(columns={"drug_id": "drug"}, inplace=True)

    merged = m[["drug", "cell_line", args.metric]].merge(
        r[["drug", "cell_line", args.metric]],
        on=["drug", "cell_line"], suffixes=("_model", "_ridge"))
    a = merged[f"{args.metric}_model"].values
    b = merged[f"{args.metric}_ridge"].values
    n = int((~(np.isnan(a) | np.isnan(b))).sum())

    if n < len(m):
        print(f"[warn] only {n} of {len(m)} model pairs matched a ridge pair — "
              "check (drug, cell_line) keys align across the two CSVs.")

    mm, mlo, mhi = bootstrap_ci(a)
    rm, rlo, rhi = bootstrap_ci(b)
    dmean, dlo, dhi = bootstrap_paired_delta(a, b)
    stat_g, p_g = paired_wilcoxon(a, b, alternative="greater")
    stat_t, p_t = paired_wilcoxon(a, b, alternative="two-sided")
    cd = cliffs_delta(a, b)

    print("=" * 70)
    print(f"  {args.label}  vs  ridge   ({args.metric}, n={n} pairs)")
    print("=" * 70)
    print(f"  {args.label:14s} {mm:.4f}  [95% CI {mlo:.4f}, {mhi:.4f}]")
    print(f"  ridge          {rm:.4f}  [95% CI {rlo:.4f}, {rhi:.4f}]")
    print(f"  Δ (model-ridge){dmean:+.4f}  [95% CI {dlo:+.4f}, {dhi:+.4f}]  (paired bootstrap)")
    print(f"  paired Wilcoxon  one-sided(model>ridge) p={p_g:.4g}   two-sided p={p_t:.4g}")
    print(f"  Cliff's delta   {cd:+.3f}")
    print("-" * 70)
    passes = (dmean >= 0.05) and (p_g < args.alpha_bonferroni)
    verdict = "PASS (Plan A)" if passes else (
        "MARGINAL" if dmean > 0 else "FAIL vs ridge")
    print(f"  GATE (need Δ>=+0.05 AND one-sided p<{args.alpha_bonferroni}):  {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
