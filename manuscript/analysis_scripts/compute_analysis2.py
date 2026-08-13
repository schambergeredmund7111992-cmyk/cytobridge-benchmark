"""
compute_analysis2.py - additional analyses for the Nature-style multi-panel
figures. All derived from the stored (27 x 3000) prediction/true vectors. No new
model runs. Outputs to manuscript/analysis/data2/.
"""
import os, sys, json
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score
RES = f"{CODE}/results"
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data2"
os.makedirs(OUT, exist_ok=True)

CONFIGS = {"loss-only": "t7_sub_loss_only", "drug-spec x1": "t7_sub_drugspec1",
           "drug-spec x3": "t7_sub_drugspec3", "drug-spec x5": "t7_sub_drugspec5",
           "low recon": "t7_sub_lamrecon01", "norm-only": "t7_sub_norm_only",
           "recovery base": "t6_sub_baseline"}


def load(tag):
    return (np.load(f"{RES}/logfc_pred_{tag}.npy"),
            np.load(f"{RES}/logfc_true_{tag}.npy"),
            pd.read_csv(f"{RES}/logfc_meta_{tag}.csv"))


def auc(pred, true, cl):
    return drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")["specificity_auc"]


pred0, true0, meta0 = load("t7_sub_loss_only")
cl0 = meta0["cell_line"].values

# build the cell-line Mean predictor (drug-invariant)
mean_pred = np.zeros_like(true0)
for c in np.unique(cl0):
    m = np.flatnonzero(cl0 == c)
    mean_pred[m] = true0[m].mean(axis=0)

# ---- 1. CALIBRATION CURVE: inject a fraction alpha of the true drug signal ----
alphas = np.linspace(0, 1, 11)
cal = []
for a in alphas:
    mix = a * true0 + (1 - a) * mean_pred
    cal.append({"alpha": float(a), "auc": float(auc(mix, true0, cl0))})
caldf = pd.DataFrame(cal)
caldf.to_csv(f"{OUT}/calibration.csv", index=False)
# where does CytoBridge-best (auc 0.583) sit on this curve?
best = 0.5833333333333334
eff_alpha = float(np.interp(best, caldf["auc"], caldf["alpha"]))
json.dump({"cytobridge_auc": best, "effective_alpha": eff_alpha},
          open(f"{OUT}/calibration_meta.json", "w"))
print(f"=== calibration: AUC(0)={cal[0]['auc']:.3f} AUC(1)={cal[-1]['auc']:.3f}; "
      f"CytoBridge eff. alpha={eff_alpha:.2f} ===")

# ---- 2. DRUG-IDENTITY CONFUSION MATRIX (one cell line) ----
cell = pd.Series(cl0).value_counts().idxmax()
idx = np.flatnonzero(cl0 == cell)
S = sorted(set().union(*[set(np.argsort(-np.abs(true0[i]))[:50]) for i in idx]))
P, T = pred0[np.ix_(idx, S)], true0[np.ix_(idx, S)]
C = np.array([[np.corrcoef(P[i], T[j])[0, 1] for j in range(len(idx))] for i in range(len(idx))])
# row-normalised assignment probability (softmax over targets)
A = np.exp(C * 6); A = A / A.sum(1, keepdims=True)
np.save(f"{OUT}/confusion_{cell}.npy", A)
diag_recovery = float(np.mean(np.argmax(C, 1) == np.arange(len(idx))))
json.dump({"cell": cell, "n": len(idx), "chance": 1 / len(idx),
           "top1_recovery": diag_recovery}, open(f"{OUT}/confusion_meta.json", "w"))
print(f"=== confusion ({cell}, n={len(idx)}): top-1 drug recovery={diag_recovery:.2f} "
      f"(chance={1/len(idx):.2f}) ===")

# ---- 3. PER-CELL-LINE metrics (collapse everywhere) ----
rows = []
for name, tag in CONFIGS.items():
    p, t, m = load(tag)
    cl = m["cell_line"].values
    for c in np.unique(cl):
        mm = np.flatnonzero(cl == c)
        a = drug_discrimination_score(p[mm], t[mm], cl[mm], top_k=50, metric="pearson")
        inter = np.mean([np.corrcoef(p[mm[i]], p[mm[j]])[0, 1]
                         for i in range(len(mm)) for j in range(i + 1, len(mm))])
        rows.append({"config": name, "cell": c, "auc": a["specificity_auc"],
                     "inter_drug_r": float(inter)})
pcl = pd.DataFrame(rows)
pcl.to_csv(f"{OUT}/per_cellline.csv", index=False)
print(f"=== per-cell-line: mean inter-drug r={pcl.inter_drug_r.mean():.3f}, "
      f"mean AUC={pcl.auc.mean():.3f} across {pcl.cell.nunique()} cells ===")

# ---- 4. PER-GENE predicted variance across drugs (collapse signature) ----
gene_std_pred, gene_std_true = [], []
for c in np.unique(cl0):
    mm = np.flatnonzero(cl0 == c)
    gene_std_pred.append(pred0[mm].std(axis=0))
    gene_std_true.append(true0[mm].std(axis=0))
gp = np.mean(gene_std_pred, axis=0); gt = np.mean(gene_std_true, axis=0)
np.savez(f"{OUT}/gene_variance.npz", pred_std=gp, true_std=gt)
print(f"=== per-gene std across drugs: predicted median={np.median(gp):.4f} "
      f"vs true median={np.median(gt):.4f} (pred ~flat = collapse) ===")

# ---- 5. MULTIPLE case studies (4 drug pairs, same cell) ----
drugs = meta0.loc[idx, "drug"].str.replace(r"\s*\(.*\)", "", regex=True).values
pairs = [(0, 1), (2, 3), (4, 5), (0, 4)]
cs = {"cell": cell}
for k, (i, j) in enumerate(pairs):
    a, b = idx[i], idx[j]
    cs[f"pair{k}"] = json.dumps({
        "A": str(drugs[i]), "B": str(drugs[j]),
        "pred_r": float(np.corrcoef(pred0[a], pred0[b])[0, 1]),
        "true_r": float(np.corrcoef(true0[a], true0[b])[0, 1])})
json.dump(cs, open(f"{OUT}/casestudies.json", "w"))
print("=== 4 case-study pairs (pred_r vs true_r) ===")
for k in range(len(pairs)):
    d = json.loads(cs[f"pair{k}"]); print(f"  {d['A']} vs {d['B']}: pred {d['pred_r']:.2f} / true {d['true_r']:.2f}")

# ---- 6. BOOTSTRAP CI for AUC and Spearman (loss-only) ----
rng = np.random.default_rng(0)
aucs, rhos = [], []
n = len(true0)
for _ in range(1000):
    bi = rng.integers(0, n, n)
    try:
        aucs.append(auc(pred0[bi], true0[bi], cl0[bi]))
    except Exception:
        pass
boot = {"auc_mean": float(np.mean(aucs)), "auc_lo": float(np.percentile(aucs, 2.5)),
        "auc_hi": float(np.percentile(aucs, 97.5))}
np.save(f"{OUT}/bootstrap_auc.npy", np.array(aucs))
json.dump(boot, open(f"{OUT}/bootstrap_meta.json", "w"))
print(f"=== bootstrap AUC (loss-only): {boot['auc_mean']:.3f} "
      f"[{boot['auc_lo']:.3f}, {boot['auc_hi']:.3f}] ===")
print("[done] ->", OUT)
