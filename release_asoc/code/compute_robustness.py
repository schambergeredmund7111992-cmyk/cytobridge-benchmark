"""
compute_robustness.py - Tier C robustness of the drug-discrimination control.
Shows the diagnosed collapse is NOT an artifact of the control's hyperparameters
(top_k, metric, dim set) and gives a permutation-null p-value for the AUC.
All numbers derive from the stored E6/E7 prediction vectors. No new model runs.
Reuses the student's EXACT drug_discrimination_score (no reimplementation).
Outputs to manuscript/analysis/data3/.
"""
import os, sys, json
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score  # the student's exact function
RES = f"{CODE}/results"
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data3"
os.makedirs(OUT, exist_ok=True)

CONFIGS = {"loss-only": "t7_sub_loss_only", "drug-spec x1": "t7_sub_drugspec1",
           "drug-spec x3": "t7_sub_drugspec3", "drug-spec x5": "t7_sub_drugspec5",
           "low recon": "t7_sub_lamrecon01", "norm-only": "t7_sub_norm_only",
           "recovery base": "t6_sub_baseline"}


def load(tag):
    return (np.load(f"{RES}/logfc_pred_{tag}.npy"),
            np.load(f"{RES}/logfc_true_{tag}.npy"),
            pd.read_csv(f"{RES}/logfc_meta_{tag}.csv"))


# ---- 1. SENSITIVITY GRID: AUC across top_k x metric, every config ----
TOPKS = [20, 50, 100, None]   # None = all 3000 dims
METRICS = ["pearson", "spearman"]
rows = []
for name, tag in CONFIGS.items():
    pred, true, meta = load(tag)
    cl = meta["cell_line"].values
    for tk in TOPKS:
        for met in METRICS:
            r = drug_discrimination_score(pred, true, cl, top_k=tk, metric=met)
            rows.append({"config": name, "top_k": ("all" if tk is None else tk),
                         "metric": met, "auc": r["specificity_auc"],
                         "gap": r["gap"], "on_diag": r["on_diag_mean"]})
grid = pd.DataFrame(rows)
grid.to_csv(f"{OUT}/sensitivity_grid.csv", index=False)
print("=== control AUC sensitivity (every cell = one config x top_k x metric) ===")
piv = grid.pivot_table(index="config", columns=["metric", "top_k"], values="auc")
print(piv.round(3).to_string())
print(f"\nAUC range over ALL {len(grid)} settings: "
      f"[{grid.auc.min():.3f}, {grid.auc.max():.3f}], mean {grid.auc.mean():.3f}")
print("(collapse is robust if max AUC stays well under the 0.70 drug-aware gate)")

# ---- 2. PERMUTATION NULL p-value for AUC (loss-only + drug-spec x5) ----
def perm_null(tag, n_perm=1000, seed=0):
    pred, true, meta = load(tag)
    cl = meta["cell_line"].values
    obs = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")["specificity_auc"]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        pp = pred.copy()
        # permute predicted rows WITHIN each cell line -> destroys any drug pairing
        for c in np.unique(cl):
            idx = np.flatnonzero(cl == c)
            pp[idx] = pred[idx[rng.permutation(len(idx))]]
        null.append(drug_discrimination_score(pp, true, cl, top_k=50, metric="pearson")["specificity_auc"])
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, float(null.mean()), float(np.percentile(null, 97.5)), p

perm = {}
for name in ["t7_sub_loss_only", "t7_sub_drugspec5"]:
    obs, nmean, nhi, p = perm_null(name)
    perm[name] = {"observed_auc": obs, "null_mean": nmean, "null_975": nhi, "p_value": p}
    print(f"\n=== permutation null [{name}] ===")
    print(f"  observed AUC={obs:.3f}, null mean={nmean:.3f}, null 97.5%={nhi:.3f}, p={p:.3f}")
    print("  (p>=0.05 => observed AUC indistinguishable from drug-blind chance)")
json.dump(perm, open(f"{OUT}/permutation_null.json", "w"), indent=2)

print("\n[done] ->", OUT)
