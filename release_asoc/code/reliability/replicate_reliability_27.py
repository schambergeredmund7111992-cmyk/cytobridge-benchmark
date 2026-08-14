"""
analysis/supp_T8/replicate_reliability_27.py
============================================
Round-3 reviewer point 2.2 — TARGET-reliability of the drug-discrimination control
restricted to the EXACT 27 scored conditions (9 held-out test drugs x 3 cell lines,
10 uM / 24 h). The 564-condition replicate ceiling (supp_T2) shows the metric rewards a
real replicate in general; the reviewer asks whether the *specific weak-signal subset we
score* is itself reliable enough that a perfect predictor could pass. If a genuine
biological replicate of these 27 conditions discriminates drugs (AUC >> 0.5), the model's
collapse (our-holdout AUC 0.4954) is a true model failure. If the replicate itself is at
chance (AUC ~ 0.5), the subset carries little distinguishable drug-specific signal at this
depth and the collapse must be down-toned to a weak-signal-subset property.

METHOD (matches the rest of the project; reuses code/eval/metrics.py::drug_discrimination_score,
NOT a re-implementation): within each (drug, cell line), split that condition's treated cells
into two random halves A/B, pseudobulk each into
    logFC = log1p(mean treated counts) - log1p(mean same-cell-line DMSO counts at 24 h),
then score half-A (as prediction) vs half-B (as truth) with the SAME off-diagonal control
(top_k=50 union of per-drug |true| genes, Pearson). Halves use ~half the cells of the real
target, so this is a CONSERVATIVE lower bound on the full-depth target reliability.

REPORTED-NUMBER OWNERSHIP: supervisor-authored (iron rule 1). Well-posedness is asserted
(Oracle=1.0, Mean~0.5) before any headline number is emitted; a 27-anchor
structure-preserving bootstrap CI, a permutation null, and split-seed stability are reported.

USAGE (l20):
  python replicate_reliability_27.py \
    --h5ad /home/zg.peng/data/guanxing/CytoBridge/code/data/processed/sciplex/sciplex_processed.h5ad \
    --code_dir /home/zg.peng/data/guanxing/CytoBridge/code --out_dir . 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The exact 9 held-out scored drugs, full names as they appear in obs['drug'].
NINE_DRUGS = [
    "AG-490 (Tyrphostin B42)",
    "Celecoxib",
    "Fulvestrant",
    "Ramelteon",
    "SL-327",
    "SRT3025 HCl",
    "Thalidomide",
    "Tofacitinib (CP-690550) Citrate",
    "Zileuton",
]
DRUG_SHORT = {  # for readable output only
    "AG-490 (Tyrphostin B42)": "AG-490",
    "SRT3025 HCl": "SRT3025",
    "Tofacitinib (CP-690550) Citrate": "Tofacitinib",
}
HIGH_DOSE = 10000.0   # 10 uM
READOUT_TIME = "24.0"
VEHICLE = "DMSO"


def _import_metric(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(f"--code_dir {code_dir} has no eval/metrics.py; point it at code/.")
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def build_replicate_halves(counts, drug, cell, dose, time, seed, min_cells=20):
    """Split each of the 27 (drug, cell line) conditions at HIGH_DOSE/READOUT_TIME into
    two random halves; return (predA, trueB, cl, drug_out) logFC pairs vs the same-cell-line
    24 h DMSO pseudobulk. Pure numpy."""
    counts = np.asarray(counts, float)
    drug = np.asarray(drug).astype(str)
    cell = np.asarray(cell).astype(str)
    dose = np.asarray(dose, float)
    time = np.asarray(time).astype(str)
    rng = np.random.default_rng(seed)

    # same-cell-line 24 h vehicle pseudobulk (log1p of mean counts)
    control = {}
    for c in np.unique(cell):
        m = (cell == c) & (drug == VEHICLE) & (dose == 0.0) & (time == READOUT_TIME)
        if m.any():
            control[c] = np.log1p(counts[m].mean(axis=0))

    predA, trueB, cl_out, drug_out, ncells = [], [], [], [], []
    for c in sorted(np.unique(cell)):
        if c not in control:
            continue
        for d in NINE_DRUGS:
            idx = np.flatnonzero(
                (cell == c) & (drug == d) & (dose == HIGH_DOSE) & (time == READOUT_TIME))
            if idx.size < min_cells:
                raise SystemExit(f"{d} x {c}: only {idx.size} treated cells (< {min_cells}).")
            idx = rng.permutation(idx)
            ha, hb = idx[: idx.size // 2], idx[idx.size // 2:]
            predA.append(np.log1p(counts[ha].mean(axis=0)) - control[c])
            trueB.append(np.log1p(counts[hb].mean(axis=0)) - control[c])
            cl_out.append(c)
            drug_out.append(DRUG_SHORT.get(d, d))
            ncells.append(idx.size)
    return (np.array(predA), np.array(trueB), np.array(cl_out),
            np.array(drug_out), np.array(ncells))


def per_anchor_auc(score_fn, pred, true, cl, top_k=50):
    """Reproduce the metric's per-anchor discrimination score a(c,d) = mean_{d'!=d}
    1[C(i,i) > C(i,d')] so we can bootstrap over the 27 anchors. Uses the metric's own
    gene-set union + Pearson via a per-cell-line call, keeping identical construction."""
    from eval.metrics import _corr_vec  # same module the metric lives in
    cl = np.asarray(cl)
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    D = pred.shape[1]
    anchors = []  # (cell, drug_idx_in_cl, a_value)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        P, T = pred[m], true[m]
        sel = set()
        for i in range(m.size):
            sel.update(np.argsort(-np.abs(T[i]))[:top_k].tolist())
        S = np.array(sorted(sel), dtype=int)
        Ps, Ts = P[:, S], T[:, S]
        mm = m.size
        C = np.array([[_corr_vec(Ps[i], Ts[j], "pearson") for j in range(mm)]
                      for i in range(mm)])
        for i in range(mm):
            diag = C[i, i]
            offs = np.array([C[i, j] for j in range(mm) if j != i], dtype=float)
            offs = offs[~np.isnan(offs)]
            if np.isnan(diag) or offs.size == 0:
                continue
            anchors.append(float(np.mean(diag > offs)))
    return np.array(anchors)


def bootstrap_ci(anchor_scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = anchor_scores.size
    boots = np.array([anchor_scores[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_null(score_fn, predA, trueB, cl, n_perm=1000, seed=0):
    """Null: within each cell line, shuffle which prediction row is paired to which drug's
    truth, recompute AUC. p = P(null AUC >= observed)."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(cl)
    obs = score_fn(predA, trueB, cl, top_k=50, metric="pearson")["specificity_auc"]
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.arange(len(cl))
        for c in np.unique(cl):
            m = np.flatnonzero(cl == c)
            perm[m] = m[rng.permutation(m.size)]
        null[b] = score_fn(predA[perm], trueB, cl, top_k=50, metric="pearson")["specificity_auc"]
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, float(null.mean()), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--code_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("."))
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--min_cells", type=int, default=20)
    args = ap.parse_args()

    score_fn = _import_metric(args.code_dir)
    import scanpy as sc  # noqa: E402
    a = sc.read_h5ad(args.h5ad)
    counts = a.layers["counts"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    drug = a.obs["drug"].values
    cell = a.obs["cell_line"].values
    dose = a.obs["dose_value"].astype(float).values
    time = a.obs["time"].astype(str).values

    # ---- per-seed replicate AUC (stability to the random half assignment) ----
    seed_rows = []
    for s in range(args.n_seeds):
        pA, tB, cl, dl, nc = build_replicate_halves(
            counts, drug, cell, dose, time, seed=s, min_cells=args.min_cells)
        r = score_fn(pA, tB, cl, top_k=50, metric="pearson")
        seed_rows.append({"seed": s, "auc": r["specificity_auc"], "gap": r["gap"],
                          "wilcoxon_p": r["wilcoxon_p_on_gt_off"], "n_anchors": r["n_pairs_scored"]})
    seed_df = pd.DataFrame(seed_rows)

    # ---- well-posedness + headline panel at seed 0 ----
    pA, tB, cl, dl, nc = build_replicate_halves(
        counts, drug, cell, dose, time, seed=0, min_cells=args.min_cells)
    mean_pred = np.zeros_like(tB)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        mean_pred[m] = tB[m].mean(axis=0)
    panel = pd.DataFrame([
        {"predictor": "Mean (neg ctrl)",
         **{k: score_fn(mean_pred, tB, cl, 50, "pearson")[k]
            for k in ("specificity_auc", "gap", "wilcoxon_p_on_gt_off")}},
        {"predictor": "Replicate A-vs-B (target reliability)",
         **{k: score_fn(pA, tB, cl, 50, "pearson")[k]
            for k in ("specificity_auc", "gap", "wilcoxon_p_on_gt_off")}},
        {"predictor": "Oracle (pos ctrl)",
         **{k: score_fn(tB.copy(), tB, cl, 50, "pearson")[k]
            for k in ("specificity_auc", "gap", "wilcoxon_p_on_gt_off")}},
    ])
    o = panel.loc[panel.predictor.str.startswith("Oracle"), "specificity_auc"].iloc[0]
    mn = panel.loc[panel.predictor.str.startswith("Mean"), "specificity_auc"].iloc[0]
    print(f"[gate] Oracle AUC={o:.4f} (must 1.0), Mean AUC={mn:.4f} (must ~0.5)")
    assert o > 0.999, "WELL-POSEDNESS FAIL: Oracle != 1.0"
    assert abs(mn - 0.5) <= 0.08, f"WELL-POSEDNESS FAIL: Mean AUC {mn} not ~0.5"

    # ---- 27-anchor structure-preserving bootstrap CI (seed 0) ----
    anchors = per_anchor_auc(score_fn, pA, tB, cl)
    ci_lo, ci_hi = bootstrap_ci(anchors, n_boot=1000, seed=0)

    # ---- permutation null (seed 0) ----
    obs, null_mean, perm_p = permutation_null(score_fn, pA, tB, cl, n_perm=1000, seed=0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out / "replicate_reliability_27_perseed.csv", index=False)
    panel.to_csv(out / "replicate_reliability_27_panel.csv", index=False)
    summary = {
        "n_conditions": int(len(cl)),
        "n_drugs": int(len(np.unique(dl))),
        "n_cell_lines": int(len(np.unique(cl))),
        "cells_per_condition_min_median_max": [int(nc.min()), int(np.median(nc)), int(nc.max())],
        "replicate_auc_seed0": round(float(obs), 4),
        "replicate_auc_mean_over_seeds": round(float(seed_df.auc.mean()), 4),
        "replicate_auc_std_over_seeds": round(float(seed_df.auc.std()), 4),
        "replicate_gap_seed0": round(float(panel.loc[1, "gap"]), 4),
        "wilcoxon_p_seed0": float(panel.loc[1, "wilcoxon_p_on_gt_off"]),
        "bootstrap_ci95_27anchor": [round(ci_lo, 4), round(ci_hi, 4)],
        "permutation_null_mean": round(float(null_mean), 4),
        "permutation_p": round(float(perm_p), 4),
        "well_posed_oracle_auc": round(float(o), 4),
        "well_posed_mean_auc": round(float(mn), 4),
        "our_holdout_model_auc_for_reference": 0.4954,
        "native_biolord_ood_auc_for_reference": 0.662,
    }
    (out / "replicate_reliability_27_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== 27-condition target-reliability panel (seed 0) ===")
    print(panel.to_string(index=False))
    print(f"\nreplicate AUC seed0 = {obs:.4f}; over {args.n_seeds} seeds "
          f"= {seed_df.auc.mean():.4f} +/- {seed_df.auc.std():.4f}")
    print(f"27-anchor bootstrap 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"permutation null mean = {null_mean:.4f}, p = {perm_p:.4f}")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] -> {out}/replicate_reliability_27_summary.json")


if __name__ == "__main__":
    main()
