"""
VALID structure-preserving bootstrap CI for the loss-only drug-discrimination AUC.

The earlier bootstrap (compute_analysis2.py section 6) resampled pair INDICES with
replacement and RECOMPUTED the cross-correlation matrix. Duplicated anchors then land
in their own off-diagonal set with an on-diagonal-valued correlation, and the strict
`diag > offs` counts that as a miss, so the AUC is biased toward 0.5 (bootstrap mean
0.507 vs the true point estimate 0.569, with the point estimate sitting at the CI's
upper edge). That is not a valid uncertainty estimate for this statistic.

This script bootstraps the 27 per-anchor discrimination scores directly (the mean of
these IS the reported AUC), which preserves the on/off-diagonal structure and centers
the bootstrap distribution on the true point estimate. Writes the corrected
bootstrap_auc.npy + bootstrap_meta.json used by Fig. 4(e).
"""
import numpy as np, pandas as pd, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data2")
RES = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results"
TAG = "t7_sub_loss_only"
pred = np.load(f"{RES}/logfc_pred_{TAG}.npy")
true = np.load(f"{RES}/logfc_true_{TAG}.npy")
meta = pd.read_csv(f"{RES}/logfc_meta_{TAG}.csv")
cl = meta["cell_line"].values
drug = meta["drug"].astype(str).values

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan

def per_anchor(pred, true, cl, top_k=50):
    """Faithful reimplementation of eval.metrics.drug_discrimination_score core loop.
    Returns per-anchor (cell, full_idx, a_i)."""
    out = []
    cl = np.asarray(cl)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P, T = pred[m], true[m]
        D = P.shape[1]
        S = np.arange(D) if (top_k is None or top_k >= D) else \
            np.array(sorted(set().union(*[set(np.argsort(-np.abs(T[i]))[:top_k].tolist()) for i in range(m.size)])), dtype=int)
        Ps, Ts = P[:, S], T[:, S]
        mm = m.size
        C = np.array([[corr(Ps[i], Ts[j]) for j in range(mm)] for i in range(mm)])
        for i in range(mm):
            diag = C[i, i]
            offs = np.array([C[i, j] for j in range(mm) if j != i], float)
            offs = offs[~np.isnan(offs)]
            if np.isnan(diag) or offs.size == 0:
                continue
            out.append((c, m[i], float(np.mean(diag > offs))))
    return out

rows = per_anchor(pred, true, cl)
a_i = np.array([r[2] for r in rows])
anchor_drug = drug[np.array([r[1] for r in rows])]
point = float(a_i.mean())
print(f"[CHECK] reproduced specificity_auc = {point:.4f}  (must match permutation observed 0.5694)")
assert abs(point - 0.5694) < 1e-3, "point estimate mismatch — pipeline broken, do NOT write artifacts"

# --- valid 1000-sample bootstrap over the per-anchor scores (seed 0, matches Fig 4e '1000-sample') ---
rng = np.random.default_rng(0)
K = len(a_i)
aucs = np.array([a_i[rng.integers(0, K, K)].mean() for _ in range(1000)])
lo, hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
boot = {"auc_mean": float(aucs.mean()), "auc_lo": lo, "auc_hi": hi,
        "point_auc": point, "n_boot": 1000, "unit": "per-anchor discrimination score",
        "note": "structure-preserving; supersedes the index-resample bootstrap that biased the AUC toward 0.5"}
np.save(f"{OUT}/bootstrap_auc.npy", aucs)
json.dump(boot, open(f"{OUT}/bootstrap_meta.json", "w"), indent=2)

# --- leave-one-drug-out SD (the paper's 'leave-one-drug-out SD') ---
udrugs = np.unique(anchor_drug)
lodo = np.array([a_i[anchor_drug != d].mean() for d in udrugs])
lodo_sd = float(lodo.std(ddof=1))
json.dump({"lodo_sd": lodo_sd, "n_drug": int(len(udrugs))}, open(f"{OUT}/lodo_meta.json", "w"), indent=2)

print(f"[VALID bootstrap] mean={boot['auc_mean']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (2dp: [{lo:.2f}, {hi:.2f}])")
print(f"                  includes 0.50? {'YES' if lo <= 0.5 <= hi else 'NO'}")
print(f"[leave-one-drug-out] SD = {lodo_sd:.4f}  (2dp {lodo_sd:.3f}), n_drug={len(udrugs)}")
print(f"[written] {OUT}/bootstrap_auc.npy, bootstrap_meta.json, lodo_meta.json")
