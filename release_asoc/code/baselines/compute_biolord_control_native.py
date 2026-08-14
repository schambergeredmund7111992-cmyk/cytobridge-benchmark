"""
compute_biolord_control_native.py - biolord control on its NATIVE sci-Plex OOD
benchmark (9 unseen drugs: Dacinostat/Givinostat/Belinostat/Hesperadin/Quisinostat/
Alvespimycin/Tanespimycin/TAK-901/Flavopiridol x A549/K562/MCF7 = 27 pairs), Option B.

Self-space scoring (pred vs biolord_true from the same pipeline); the vectors are already
aligned with biolord_vectors/meta.csv row-for-row, so NO reindex-to-our-9-drugs is needed
(unlike compute_biolord_control.py, which targets our sci-Plex holdout drug set).
Reuses the EXACT student control function. Well-posedness gate: Oracle=1.0, Mean~=0.5.
"""
import os, sys, json
import numpy as np, pandas as pd
CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score
VEC = sys.argv[1] if len(sys.argv) > 1 else "biolord_vectors"
TAG = "ours" if "ours" in VEC else "native"
CV = f"/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/{VEC}"
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data_biolord"
os.makedirs(OUT, exist_ok=True)
print(f"=== biolord control [{TAG}] from {VEC} ===")

meta = pd.read_csv(f"{CV}/meta.csv")
pred_t = np.load(f"{CV}/pred_treated_pb.npy")
true_t = np.load(f"{CV}/true_treated_pb.npy")
ctrl = np.load(f"{CV}/control_pb.npy")
cl = meta["cell_line"].astype(str).values
print(f"biolord native: pred {pred_t.shape}, {meta.drug.nunique()} drugs x {len(np.unique(cl))} cells")
print("ood drugs:", sorted(meta.drug.unique().tolist()))

mx = max(float(pred_t.max()), float(true_t.max()))
if mx > 30:
    pred_logfc = np.log1p(np.clip(pred_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    true_logfc = np.log1p(np.clip(true_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    print("[space] raw-count pseudobulks -> logFC = log1p(treated)-log1p(control)")
else:
    pred_logfc = pred_t - ctrl
    true_logfc = true_t - ctrl
    print(f"[space] log-normalized pseudobulks (max {mx:.2f}) -> logFC = treated - control")


def panel(pred, true, cl, label):
    r50 = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")
    rall = drug_discrimination_score(pred, true, cl, top_k=None, metric="spearman")
    inter = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                inter.append(np.corrcoef(pred[m[i]], pred[m[j]])[0, 1])
    return {"predictor": label, "auc_deg50": round(float(r50["specificity_auc"]), 4),
            "gap_deg50": round(float(r50["gap"]), 4), "on_diag_deg50": round(float(r50["on_diag_mean"]), 4),
            "wilcoxon_p_on_gt_off": round(float(r50["wilcoxon_p_on_gt_off"]), 4),
            "auc_all_spearman": round(float(rall["specificity_auc"]), 4),
            "inter_drug_pearson": round(float(np.mean(inter)), 4) if inter else np.nan}


oracle = panel(true_logfc.copy(), true_logfc, cl, "Oracle (pos ctrl)")
mean_pred = np.zeros_like(true_logfc)
for c in np.unique(cl):
    m = np.flatnonzero(cl == c)
    mean_pred[m] = true_logfc[m].mean(axis=0)
mean_row = panel(mean_pred, true_logfc, cl, "Mean (neg ctrl)")
rng = np.random.default_rng(0)
rand_row = panel(true_logfc[rng.permutation(len(true_logfc))], true_logfc, cl, "Random (neg ctrl)")
print(f"[gate] Oracle AUC={oracle['auc_deg50']} (must be 1.0), Mean AUC={mean_row['auc_deg50']} (must be ~0.5)")
if oracle["auc_deg50"] < 0.999:
    raise SystemExit("WELL-POSEDNESS FAIL: Oracle != 1.0 -> biolord true/units broken.")
if abs(mean_row["auc_deg50"] - 0.5) > 0.08:
    print(f"[warn] Mean AUC {mean_row['auc_deg50']} deviates from 0.50 by >0.08.")

bio = panel(pred_logfc, true_logfc, cl, "biolord (self-space, native OOD)")
rows = [rand_row, mean_row, bio, oracle]
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/biolord_control_panel_{TAG}.csv", index=False)
with open(f"{OUT}/biolord_summary_{TAG}.json", "w") as fh:
    json.dump({"biolord": bio, "oracle": oracle, "mean": mean_row, "random": rand_row,
               "n_pairs": int(len(cl)), "ood_drugs": sorted(meta.drug.unique().tolist())},
              fh, indent=2, default=float)
print("\n=== biolord drug-discrimination control (self-space, native sci-Plex OOD) ===")
print(df.round(3).to_string(index=False))
print(f"\nHEADLINE: biolord AUC@50(DEG,pearson) = {bio['auc_deg50']}, "
      f"inter-drug pearson = {bio['inter_drug_pearson']}")
print(f"[done] -> {OUT}/biolord_control_panel.csv")
