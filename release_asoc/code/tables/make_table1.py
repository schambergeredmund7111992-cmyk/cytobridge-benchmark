"""
make_table1.py - regenerate the main per-configuration table (Table 1) from a SINGLE
documented recipe, fixing review findings F02/F16. The previous manuscript Table 1
reported Spearman values (0.133/0.132/.../0.059) and a Mean baseline 0.357 that are
not reproducible from any released artifact (they match neither on-diagonal
Spearman@50 nor on-diagonal Pearson@50 computed from the E6/E7 per-config vectors).

RECIPE (documented, reproducible):
  standard per-pair correlation = on-diagonal mean of the student control function,
  drug_discrimination_score(pred, true, cell_lines, top_k=50, metric=...)["on_diag_mean"],
  i.e. mean over the 27 (drug, cell line) pairs of corr(pred_pair, true_pair) on the
  top-50 differentially expressed genes. We report Spearman@50 (primary) and
  Pearson@50, plus the off-diagonal control AUC@50 (Pearson) and the inter-drug
  prediction Pearson (collapse signature).

SPACES: the 7 CytoBridge configs and the Mean baseline are scored in the E6/E7
canonical reconstruction space (logfc_*_<config>.npy). Ridge lives in its own clean
9-drug holdout reconstruction (baseline_vectors_sub/) and is reported with a space
footnote, because its reconstruction correlates only ~0.26 with the canonical one
(lost dose-pooling/control provenance). Oracle = 1.0 and Mean ~= 0.50 control AUC in
every consistent space, the well-posedness anchors.

Output: data/table1_regenerated.csv  (every Table 1 number traces here).
"""
import os
import sys
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score
RES = f"{CODE}/results"
HERE = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis"
BV = f"{HERE}/baseline_vectors_sub"
OUT = f"{HERE}/data"

CONFIGS = [
    ("loss-only", "t7_sub_loss_only"),
    ("drug-spec x1", "t7_sub_drugspec1"),
    ("drug-spec x3", "t7_sub_drugspec3"),
    ("drug-spec x5", "t7_sub_drugspec5"),
    ("low recon", "t7_sub_lamrecon01"),
    ("norm-only", "t7_sub_norm_only"),
    ("recovery base", "t6_sub_baseline"),
]


def row_for(pred, true, cl, label, space):
    sp = drug_discrimination_score(pred, true, cl, top_k=50, metric="spearman")
    pe = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")
    # per-pair (on-diagonal) median Pearson@50 for the "median Pearson" prose value
    inter = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                inter.append(np.corrcoef(pred[m[i]], pred[m[j]])[0, 1])
    return {
        "predictor": label, "space": space,
        "spearman50_ondiag": round(float(sp["on_diag_mean"]), 4),
        "pearson50_ondiag": round(float(pe["on_diag_mean"]), 4),
        "control_auc50": round(float(pe["specificity_auc"]), 4),
        "control_gap50": round(float(pe["gap"]), 4),
        "inter_drug_pearson": round(float(np.mean(inter)), 4) if inter else np.nan,
    }


rows = []
# 7 CytoBridge configs in E6/E7 canonical space
for label, tag in CONFIGS:
    p = np.load(f"{RES}/logfc_pred_{tag}.npy")
    t = np.load(f"{RES}/logfc_true_{tag}.npy")
    cl = pd.read_csv(f"{RES}/logfc_meta_{tag}.csv")["cell_line"].values
    rows.append(row_for(p, t, cl, label, "E6E7"))

# Mean baseline in E6/E7 canonical space (cell-line average of the canonical truth)
t = np.load(f"{RES}/logfc_true_t7_sub_loss_only.npy")
cl = pd.read_csv(f"{RES}/logfc_meta_t7_sub_loss_only.csv")["cell_line"].values
mean = np.zeros_like(t)
for c in np.unique(cl):
    idx = np.flatnonzero(cl == c)
    mean[idx] = t[idx].mean(0)
rows.append(row_for(mean, t, cl, "Mean (cell-line avg)", "E6E7"))

# Ridge in its own clean 9-drug holdout space
pr = np.load(f"{BV}/pred_ridge.npy")
tr = np.load(f"{BV}/true.npy")
clr = pd.read_csv(f"{BV}/meta.csv")["cell_line"].values
ridge = row_for(pr, tr, clr, "Ridge (clean holdout)", "ridge_recon")
# median per-pair Pearson@50 for Ridge (the manuscript's 0.21 prose value)
med = []
for i in range(len(tr)):
    o = np.argsort(np.abs(tr[i]))[::-1][:50]
    med.append(np.corrcoef(pr[i][o], tr[i][o])[0, 1])
ridge["median_perpair_pearson50"] = round(float(np.median(med)), 4)
rows.append(ridge)

# chemCPA in its own clean reconstruction (log-normalized, logFC = treated - control)
CV = f"{HERE}/chemcpa_vectors"
cpred = np.load(f"{CV}/pred_treated_pb.npy")
ctrue = np.load(f"{CV}/true_treated_pb.npy")
cctrl3 = np.load(f"{CV}/control_pb.npy")
cmeta = pd.read_csv(f"{CV}/meta.csv")
cclc = cmeta["cell_line"].values
_cells = ["A549", "K562", "MCF7"]
cctrl = np.stack([cctrl3[_cells.index(c)] for c in cclc])
rows.append(row_for(cpred - cctrl, ctrue - cctrl, cclc, "chemCPA (clean holdout)", "chemcpa_recon"))

# per-space Mean baseline Spearman@50, for the caption's "loses to its own Mean" point
def _mean_spear(true, cl):
    m = np.zeros_like(true)
    for c in np.unique(cl):
        idx = np.flatnonzero(cl == c)
        m[idx] = true[idx].mean(0)
    return round(float(drug_discrimination_score(m, true, cl, top_k=50, metric="spearman")["on_diag_mean"]), 4)

space_means = {
    "E6E7": _mean_spear(np.load(f"{RES}/logfc_true_t7_sub_loss_only.npy"),
                        pd.read_csv(f"{RES}/logfc_meta_t7_sub_loss_only.csv")["cell_line"].values),
    "ridge_recon": _mean_spear(tr, clr),
    "chemcpa_recon": _mean_spear(ctrue - cctrl, cclc),
}
pd.DataFrame([{"space": k, "mean_spearman50": v} for k, v in space_means.items()]).to_csv(
    f"{OUT}/per_space_mean_spearman.csv", index=False)
print("per-space Mean Spearman@50:", space_means)

tab = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
tab.to_csv(f"{OUT}/table1_regenerated.csv", index=False)
print(tab.to_string(index=False))
print(f"\n[done] -> {OUT}/table1_regenerated.csv")
print("\nKEY CONTRAST: all 7 configs' standard per-pair Spearman@50 vs Mean baseline; "
      "control AUC ~0.5 throughout.")
print("Mean Spearman@50 =", tab.loc[tab.predictor.str.startswith('Mean'), 'spearman50_ondiag'].iloc[0],
      "| best config Spearman@50 =", tab.loc[tab.space == 'E6E7', 'spearman50_ondiag'].iloc[:7].max())
