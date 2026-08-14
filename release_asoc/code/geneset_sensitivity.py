"""Gene-set sensitivity for the off-diagonal control (reviewer round-2, Major 2).
Reproduces the loss-only discrimination AUC under two gene-set rules:
  union    = per-cell-line union of per-drug top-50 |y| genes (the reported primary)
  matched  = each anchor's own top-50 |y| genes for both on- and off-diagonal
Writes data3/geneset_sensitivity.json (values quoted in Section 3.3)."""
import numpy as np, pandas as pd, json, os
RES = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results"
TAG = "t7_sub_loss_only"
pred = np.load(f"{RES}/logfc_pred_{TAG}.npy"); true = np.load(f"{RES}/logfc_true_{TAG}.npy")
cl = pd.read_csv(f"{RES}/logfc_meta_{TAG}.csv")["cell_line"].values
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan
def auc(P, T, cl, mode, top_k=50):
    cl = np.asarray(cl); scores = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2: continue
        Pm, Tm = P[m], T[m]
        if mode == "union":
            S = np.array(sorted(set().union(*[set(np.argsort(-np.abs(Tm[i]))[:top_k]) for i in range(m.size)])))
            for i in range(m.size):
                on = corr(Pm[i, S], Tm[i, S]); off = [corr(Pm[i, S], Tm[j, S]) for j in range(m.size) if j != i]
                scores.append(np.mean([on > o for o in off if not np.isnan(o)]))
        else:  # matched
            for i in range(m.size):
                Si = np.argsort(-np.abs(Tm[i]))[:top_k]
                on = corr(Pm[i, Si], Tm[i, Si]); off = [corr(Pm[i, Si], Tm[j, Si]) for j in range(m.size) if j != i]
                scores.append(np.mean([on > o for o in off if not np.isnan(o)]))
    return float(np.mean(scores))
out = {"config": TAG, "top_k": 50, "metric": "pearson",
       "auc_union_top50": round(auc(pred, true, cl, "union"), 4),
       "auc_matched_top50": round(auc(pred, true, cl, "matched"), 4)}
os.makedirs("data3", exist_ok=True)
json.dump(out, open("data3/geneset_sensitivity.json", "w"), indent=2)
print(json.dumps(out, indent=2))
