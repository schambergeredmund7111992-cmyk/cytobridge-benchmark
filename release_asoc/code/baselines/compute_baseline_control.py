"""
compute_baseline_control.py - the headline supplementary experiment.
Computes the off-diagonal drug-discrimination control for SIMPLE BASELINES
(Mean, Ridge[, scGPT-zs]) on the SAME test split as CytoBridge, so we can show
whether a linear fingerprint baseline discriminates drugs where the cross-modal
foundation bridge collapses.

The baseline PREDICTION vectors are produced on cityu and copied to
  manuscript/analysis/baseline_vectors/{pred_mean,pred_ridge,true,meta}.*
This script (supervisor-owned, reuses the student's EXACT control function) does
the number-critical part: alignment verification + control AUC. No reimplementation
of the metric.

If the baseline vectors are not present yet, it runs a DRY RUN that treats each
CytoBridge config's own prediction as a pseudo-baseline, reproducing its known AUC
to validate the alignment + scoring pipeline end to end.
Outputs to manuscript/analysis/data_baseline/.
"""
import os, sys, json
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score
RES = f"{CODE}/results"
BV = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/baseline_vectors_sub"  # clean 9-drug holdout (Ridge trained on other 174 drugs)
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data_baseline"
os.makedirs(OUT, exist_ok=True)

# ---- canonical test ground truth + ordering (from a CytoBridge config) ----
CANON = "t7_sub_loss_only"
true_canon = np.load(f"{RES}/logfc_true_{CANON}.npy")
meta_canon = pd.read_csv(f"{RES}/logfc_meta_{CANON}.csv")
key_canon = (meta_canon["drug"].astype(str) + "||" + meta_canon["cell_line"].astype(str)).values
cl_canon = meta_canon["cell_line"].values
print(f"canonical test: {true_canon.shape}, {len(np.unique(cl_canon))} cell lines, "
      f"{meta_canon['drug'].nunique()} drugs")


def control_panel(pred, true, cl, label):
    """Score one predictor with the student's exact function, top-50 DEG pearson
    (primary) plus the spearman-all sensitivity point flagged in Tier C."""
    r50 = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")
    rall = drug_discrimination_score(pred, true, cl, top_k=None, metric="spearman")
    # inter-drug pearson of the PREDICTIONS (collapse signature)
    inter = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                inter.append(np.corrcoef(pred[m[i]], pred[m[j]])[0, 1])
    return {"predictor": label, "auc_deg50": r50["specificity_auc"],
            "gap_deg50": r50["gap"], "on_diag_deg50": r50["on_diag_mean"],
            "auc_all_spearman": rall["specificity_auc"],
            "inter_drug_pearson": float(np.mean(inter)) if inter else np.nan}


def align(pred_b, meta_b):
    """Reindex baseline rows to the canonical (drug, cell_line) order."""
    key_b = (meta_b["drug"].astype(str) + "||" + meta_b["cell_line"].astype(str)).values
    pos = {k: i for i, k in enumerate(key_b)}
    missing = [k for k in key_canon if k not in pos]
    if missing:
        raise SystemExit(f"baseline missing {len(missing)} canonical pairs, e.g. {missing[:3]}")
    order = [pos[k] for k in key_canon]
    return pred_b[order]


rows = []
have_baselines = os.path.exists(f"{BV}/pred_ridge.npy")

if have_baselines:
    print("\n=== REAL baseline vectors found -> scoring panel ===")
    true_b = np.load(f"{BV}/true.npy")
    meta_b = pd.read_csv(f"{BV}/meta.csv")
    # INVARIANT 1: baseline 'true' must match canonical true after alignment
    true_b_al = align(true_b, meta_b)
    dev = np.abs(true_b_al - true_canon).max()
    corr = np.corrcoef(true_b_al.ravel(), true_canon.ravel())[0, 1]
    print(f"  [invariant] true alignment: max|dev|={dev:.4g}, corr={corr:.5f} "
          f"(must be ~1.0; else gene/pair ordering differs)")
    if corr < 0.99:
        raise SystemExit("ALIGNMENT FAIL: baseline true does not match canonical true.")
    for tag, label in [("pred_mean", "Mean"), ("pred_ridge", "Ridge"),
                       ("pred_scgpt_zs", "scGPT-zs")]:
        f = f"{BV}/{tag}.npy"
        if not os.path.exists(f):
            continue
        pb = align(np.load(f), meta_b)
        rows.append(control_panel(pb, true_canon, cl_canon, label))
else:
    print("\n=== DRY RUN (baseline vectors absent) -> pipeline self-test on configs ===")
    for tag, label in [("t7_sub_loss_only", "CB:loss-only(self-test)"),
                       ("t7_sub_drugspec5", "CB:drug-spec x5(self-test)")]:
        pb = np.load(f"{RES}/logfc_pred_{tag}.npy")
        rows.append(control_panel(pb, true_canon, cl_canon, label))

# positive/negative controls always included (computed locally, no external vectors)
mean_pred = np.zeros_like(true_canon)
for c in np.unique(cl_canon):
    m = np.flatnonzero(cl_canon == c)
    mean_pred[m] = true_canon[m].mean(axis=0)
rng = np.random.default_rng(0)
rand_pred = true_canon[rng.permutation(len(true_canon))]
rows.append(control_panel(mean_pred, true_canon, cl_canon, "Mean (neg ctrl)"))
rows.append(control_panel(rand_pred, true_canon, cl_canon, "Random (neg ctrl)"))
rows.append(control_panel(true_canon.copy(), true_canon, cl_canon, "Oracle (pos ctrl)"))

panel = pd.DataFrame(rows)
panel.to_csv(f"{OUT}/baseline_control_panel.csv", index=False)
print("\n=== drug-discrimination control panel ===")
print(panel.round(3).to_string(index=False))
print(f"\n[done] real_baselines={have_baselines} ->", OUT)
