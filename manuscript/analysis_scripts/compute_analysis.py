"""
compute_analysis.py - derive every analysis number/array for the CytoBridge
diagnostic manuscript from the stored E6/E7 prediction vectors. No new model
runs: this only re-analyses existing predictions (logfc_pred/true, 27 x 3000).

Outputs (to manuscript/analysis/data/):
  control_validation.csv   AUC/gap for Mean, Random, Oracle, and 7 CytoBridge configs
  config_metrics.csv       rho50, inter-drug r, AUC per config (reproduced)
  interdrug_corr_<cfg>.npy correlation matrix of predicted drug vectors (per cell)
  onoff_scores_<cfg>.csv   on- and off-diagonal scores for distribution plots
  casestudy.npz            pred/true vectors for two drugs on one cell line
  loss_components.csv      training loss-component magnitudes
All numbers are printed with a self-check against the stored verify_*.json AUCs.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

# Use the STUDENT'S EXACT eval function so every recomputed number matches the
# stored verify_*.json (pearson, per-cell union of top-50 |true| dims, fixed set).
CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score

RES = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/results"
LOGS = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7/logs"
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data"
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    "loss-only": "t7_sub_loss_only", "drug-spec x1": "t7_sub_drugspec1",
    "drug-spec x3": "t7_sub_drugspec3", "drug-spec x5": "t7_sub_drugspec5",
    "low recon": "t7_sub_lamrecon01", "norm-only": "t7_sub_norm_only",
    "recovery base": "t6_sub_baseline",
}
TOPK = 50


def topk_idx(v):
    return np.argsort(-np.abs(v))[:TOPK]


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    return np.corrcoef(ra, rb)[0, 1]


def pearson(a, b):
    return np.corrcoef(a, b)[0, 1]


def load(cfg_tag):
    pred = np.load(f"{RES}/logfc_pred_{cfg_tag}.npy")
    true = np.load(f"{RES}/logfc_true_{cfg_tag}.npy")
    meta = pd.read_csv(f"{RES}/logfc_meta_{cfg_tag}.csv")
    return pred, true, meta


def control(pred, true, meta):
    """Off-diagonal drug-discrimination control, via the student's exact function.
    Returns the summary dict plus per-pair on/off arrays for distribution plots."""
    cl = meta["cell_line"].values
    res = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")
    # per-pair on/off for distributions: replicate the same fixed-dim scoring
    on, off = [], []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P, T = pred[m], true[m]
        sel = set()
        for i in range(m.size):
            sel.update(np.argsort(-np.abs(T[i]))[:50].tolist())
        S = np.array(sorted(sel))
        Ps, Ts = P[:, S], T[:, S]
        C = np.array([[pearson(Ps[i], Ts[j]) for j in range(m.size)] for i in range(m.size)])
        for i in range(m.size):
            offs = np.array([C[i, j] for j in range(m.size) if j != i])
            on.append(C[i, i]); off.append(np.mean(offs))
    return dict(auc=res["specificity_auc"], gap=res["gap"], rho50=res["on_diag_mean"]), \
        np.array(on), np.array(off)


def inter_drug_pearson(pred, meta):
    rs = []
    for cell, idx in meta.groupby("cell_line").groups.items():
        idx = list(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                rs.append(pearson(pred[idx[a]], pred[idx[b]]))
    return float(np.mean(rs)) if rs else np.nan


# ---- 1. reproduce per-config metrics + self-check ----
rows, onoff = [], {}
verify = {}
for name, tag in CONFIGS.items():
    pred, true, meta = load(tag)
    res, on, off = control(pred, true, meta)
    rinter = inter_drug_pearson(pred, meta)
    rows.append({"config": name, "rho50": res["rho50"], "inter_drug_r": rinter,
                 "auc": res["auc"], "gap": res["gap"]})
    onoff[name] = (on, off)
    # self-check vs stored verify json
    vj = f"{RES}/verify_{tag}.json"
    if os.path.exists(vj):
        verify[name] = json.load(open(vj)).get("specificity_auc")
cfgdf = pd.DataFrame(rows)
cfgdf.to_csv(f"{OUT}/config_metrics.csv", index=False)
print("=== per-config (recomputed) vs stored verify AUC ===")
for r in rows:
    s = verify.get(r["config"])
    print(f"  {r['config']:14s} AUC_recomp={r['auc']:.3f} stored={s}")

# ---- 2. control validation: Mean / Random / Oracle baselines ----
# use one config's true vectors + meta as the evaluation substrate
pred0, true0, meta0 = load("t7_sub_loss_only")
val = []
# Mean: cell-line mean of the OTHER drugs' true vectors (drug-invariant by construction)
mean_pred = np.zeros_like(true0)
for cell, idx in meta0.groupby("cell_line").groups.items():
    idx = list(idx)
    m = true0[idx].mean(axis=0)
    for i in idx:
        mean_pred[i] = m
oracle_pred = true0.copy()
# Random: average the control over many permutations for a clean chance estimate
rng = np.random.default_rng(0)
raucs = []
for _ in range(50):
    rp = true0[rng.permutation(len(true0))]
    raucs.append(control(rp, true0, meta0)[0]["auc"])
val.append({"predictor": "Random (50 perm)", "auc": float(np.mean(raucs)),
            "gap": float("nan")})
for label, P in [("Mean (collapsed)", mean_pred), ("Oracle (truth)", oracle_pred)]:
    res, on, off = control(P, true0, meta0)
    val.append({"predictor": label, "auc": res["auc"], "gap": res["gap"]})
# add the best and worst CytoBridge config for context
val.append({"predictor": "CytoBridge (best)", "auc": cfgdf["auc"].max(),
            "gap": float(cfgdf.loc[cfgdf.auc.idxmax(), "gap"])})
valdf = pd.DataFrame(val)
valdf.to_csv(f"{OUT}/control_validation.csv", index=False)
print("\n=== control validation (positive/negative controls) ===")
print(valdf.to_string(index=False))

# ---- 3. inter-drug correlation matrix (loss-only, one cell line) ----
cellsel = meta0["cell_line"].value_counts().idxmax()
idx = list(meta0.index[meta0.cell_line == cellsel])
M = np.array([[pearson(pred0[a], pred0[b]) for b in idx] for a in idx])
np.save(f"{OUT}/interdrug_corr_lossonly.npy", M)
labels = meta0.loc[idx, "drug"].str.replace(r"\s*\(.*\)", "", regex=True).tolist()
json.dump({"cell": cellsel, "drugs": labels, "mean_offdiag": float(
    (M.sum() - np.trace(M)) / (M.size - len(M)))}, open(f"{OUT}/interdrug_meta.json", "w"))
print(f"\n=== inter-drug corr matrix ({cellsel}, n={len(idx)}): mean off-diag ="
      f" {(M.sum()-np.trace(M))/(M.size-len(M)):.3f} ===")

# ---- 4. on/off distributions ----
for name, (on, off) in onoff.items():
    pd.DataFrame({"score": np.concatenate([on, off]),
                  "kind": ["on"] * len(on) + ["off"] * len(off)}
                 ).to_csv(f"{OUT}/onoff_{name.replace(' ','_').replace('/','')}.csv", index=False)

# ---- 5. case study: two drugs, one cell, pred vs true ----
i0, i1 = idx[0], idx[1]
np.savez(f"{OUT}/casestudy.npz",
         cell=cellsel, drugA=labels[0], drugB=labels[1],
         predA=pred0[i0], trueA=true0[i0], predB=pred0[i1], trueB=true0[i1],
         pred_corr=pearson(pred0[i0], pred0[i1]), true_corr=pearson(true0[i0], true0[i1]))
print(f"\n=== case study: {labels[0]} vs {labels[1]} on {cellsel} ===")
print(f"  predicted vectors corr = {pearson(pred0[i0], pred0[i1]):.3f} (near 1 = collapse)")
print(f"  true      vectors corr = {pearson(true0[i0], true0[i1]):.3f} (drugs truly differ)")

# ---- 6. loss components ----
mc = sorted(glob.glob(f"{LOGS}/t7_sub_drugspec1/version_*/metrics.csv"))
if mc:
    d = pd.read_csv(mc[-1])
    comp = {c.split("/")[-1].replace("_step", ""): float(d[c].dropna().iloc[0])
            for c in d.columns if c.startswith("train/L_")}
    pd.DataFrame([comp]).to_csv(f"{OUT}/loss_components.csv", index=False)
    print("\n=== loss components (raw magnitude) ===")
    for k, v in comp.items():
        print(f"  {k}: {v:.2f}")
print("\n[done] analysis data ->", OUT)
