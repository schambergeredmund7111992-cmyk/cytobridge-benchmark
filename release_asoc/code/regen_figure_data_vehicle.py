"""Regenerate every figure-backing data file under the leak-free per-cell-line vehicle.

Only the CytoBridge-derived quantities change; rows for Ridge, chemCPA, biolord and the
Mean, which were already scored against one vehicle per cell line, are carried through
untouched. Gated on a known-answer rebuild of the stored truth arrays.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

E6E7 = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, E6E7)
from eval.metrics import drug_discrimination_score as dds
from eval.metrics import inter_drug_pearson

SPL = ("/Users/cgxmac/Desktop/CytoBridge/student_progress_CytoBridge_收尾交接_赵希宸/"
       "汇报/成果/X1_canonical_split/splits_canonical")
RES = f"{E6E7}/results"
MD = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis"
D1, D2 = f"{MD}/data", f"{MD}/data2"

CONFIGS = [("loss-only", "t7_sub_loss_only"), ("drug-spec x1", "t7_sub_drugspec1"),
           ("drug-spec x3", "t7_sub_drugspec3"), ("drug-spec x5", "t7_sub_drugspec5"),
           ("low recon", "t7_sub_lamrecon01"), ("norm-only", "t7_sub_norm_only"),
           ("recovery base", "t6_sub_baseline")]
SHORT = {"AG-490 (Tyrphostin B42)": "AG-490", "Tofacitinib (CP-690550) Citrate": "Tofacitinib Citrate"}

# ------------------------------------------------------------------ frozen pair table
man = pd.read_parquet(f"{SPL}/sciplex_test.parquet")
tre = np.load(f"{SPL}/sciplex_test_treated_counts.npy", mmap_mode="r")
ctl = np.load(f"{SPL}/sciplex_test_control_counts.npy", mmap_mode="r")
drug_r, cell_r = man["drug_id"].astype(str).to_numpy(), man["cell_line"].astype(str).to_numpy()
keys, cp, tp = [], [], []
for (d, c), g in pd.DataFrame({"drug": drug_r, "cell": cell_r}).groupby(["drug", "cell"]):
    i = np.asarray(list(g.index), dtype=int)
    keys.append((d, c)); cp.append(np.asarray(ctl[i]).mean(0)); tp.append(np.asarray(tre[i]).mean(0))
cp, tp = np.stack(cp), np.stack(tp)
cl = np.array([c for _, c in keys]); dr = np.array([d for d, _ in keys])
cline = np.stack([np.asarray(ctl[cell_r == c]).mean(0) for c in cl])
SHIFT = np.log1p(cp) - np.log1p(cline)
TRUE = np.log1p(tp) - np.log1p(cline)
CELLS = ["A549", "K562", "MCF7"]


def load(stem):
    pr = np.load(f"{RES}/logfc_pred_{stem}.npy"); tu = np.load(f"{RES}/logfc_true_{stem}.npy")
    mt = pd.read_csv(f"{RES}/logfc_meta_{stem}.csv")
    inv = np.argsort([keys.index((r.drug, r.cell_line)) for r in mt.itertuples()])
    assert np.abs((np.log1p(tp) - np.log1p(cp)) - tu[inv]).max() < 1e-5, f"gate fail {stem}"
    return pr[inv] + SHIFT, tu[inv] + SHIFT


P = {name: load(stem)[0] for name, stem in CONFIGS}
print(f"known-answer gate PASS for all {len(P)} configurations")

# The cell-line Mean is the collapse reference. It is built here rather than further down
# because table1_regenerated.csv must carry its own row: the figure scripts previously had no
# Mean row to read and hard-coded the value, which is how fig_collapse_overview.pdf came to be
# unreproducible from any script on disk.
mean_pred = np.zeros_like(TRUE)
for c in CELLS:
    m = cl == c
    mean_pred[m] = TRUE[m].mean(0)

# ------------------------------------------------------------------ D1/config_metrics.csv
rows, t1_new = [], []
for name, _ in CONFIGS:
    p = P[name]
    r50 = dds(p, TRUE, cl, top_k=50, metric="pearson")
    s50 = dds(p, TRUE, cl, top_k=50, metric="spearman")
    rows.append(dict(config=name, rho50=r50["on_diag_mean"], inter_drug_r=inter_drug_pearson(p, cl),
                     auc=r50["specificity_auc"], gap=r50["gap"]))
    t1_new.append(dict(predictor=name, space="E6E7", spearman50_ondiag=round(s50["on_diag_mean"], 4),
                       pearson50_ondiag=round(r50["on_diag_mean"], 4),
                       control_auc50=round(r50["specificity_auc"], 4),
                       control_gap50=round(r50["gap"], 4),
                       inter_drug_pearson=round(inter_drug_pearson(p, cl), 4),
                       median_perpair_pearson50=np.nan))
_mr = dds(mean_pred, TRUE, cl, top_k=50, metric="pearson")
_ms = dds(mean_pred, TRUE, cl, top_k=50, metric="spearman")
t1_new.append(dict(predictor="Mean (cell-line avg)", space="E6E7",
                   spearman50_ondiag=round(_ms["on_diag_mean"], 4),
                   pearson50_ondiag=round(_mr["on_diag_mean"], 4),
                   control_auc50=round(_mr["specificity_auc"], 4),
                   control_gap50=round(_mr["gap"], 4),
                   inter_drug_pearson=round(inter_drug_pearson(mean_pred, cl), 4),
                   median_perpair_pearson50=np.nan))
pd.DataFrame(rows).to_csv(f"{D1}/config_metrics.csv", index=False)

t1 = pd.read_csv(f"{D1}/table1_regenerated.csv")
keep = t1[t1["space"] != "E6E7"]
pd.concat([pd.DataFrame(t1_new), keep], ignore_index=True).to_csv(f"{D1}/table1_regenerated.csv", index=False)
print(f"config_metrics.csv + table1_regenerated.csv  (kept {len(keep)} non-E6E7 rows untouched)")

# ------------------------------------------------------------------ D1/control_validation.csv
mean_pred = np.zeros_like(TRUE)
for c in CELLS:
    m = cl == c
    mean_pred[m] = TRUE[m].mean(0)
rng = np.random.default_rng(0)
rand = []
for _ in range(50):
    q = TRUE.copy()
    for c in CELLS:
        m = np.flatnonzero(cl == c)
        q[m] = TRUE[rng.permutation(m)]
    rand.append(dds(q, TRUE, cl, top_k=50, metric="pearson")["specificity_auc"])
best = max(dds(P[n], TRUE, cl, top_k=50, metric="pearson")["specificity_auc"] for n, _ in CONFIGS)
best_name = max(CONFIGS, key=lambda kv: dds(P[kv[0]], TRUE, cl, top_k=50, metric="pearson")["specificity_auc"])[0]
oracle = dds(TRUE, TRUE, cl, top_k=50, metric="pearson")
mean_r = dds(mean_pred, TRUE, cl, top_k=50, metric="pearson")
blind = np.log1p(np.tile(tp.mean(0), (len(keys), 1))) - np.log1p(cline)
blind_r = dds(blind, TRUE, cl, top_k=50, metric="pearson")
# The three external baselines are NOT recomputed here: each is scored in its own clean
# reconstruction by compute_baseline_control.py / compute_chemcpa_control.py /
# compute_biolord_control.py, whose panels carry the precise AUC and gap. Carry those rows
# through untouched, exactly as the non-E6E7 rows of table1_regenerated.csv are carried through.
# (They used to be overwritten with hard-coded constants here, which silently dropped the gaps
# and coarsened Ridge 0.509 -> 0.51 every time this script was re-run.)
_EXTERNAL = ("Ridge", "chemCPA", "biolord")
_prev = pd.read_csv(f"{D1}/control_validation.csv")
_carry = _prev[_prev["predictor"].str.startswith(_EXTERNAL)]
assert len(_carry) == len(_EXTERNAL), (
    f"expected one carried row per external baseline, found {list(_carry['predictor'])}; "
    "re-run compute_{baseline,chemcpa,biolord}_control.py before this script")
pd.concat([pd.DataFrame([
    dict(predictor="Random (50 perm)", auc=float(np.mean(rand)), gap=np.nan),
    dict(predictor="Mean (collapsed)", auc=mean_r["specificity_auc"], gap=mean_r["gap"]),
    dict(predictor="No-drug-information", auc=blind_r["specificity_auc"], gap=blind_r["gap"]),
]), _carry, pd.DataFrame([
    dict(predictor="CytoBridge (best)", auc=best, gap=np.nan),
    dict(predictor="Oracle (truth)", auc=oracle["specificity_auc"], gap=oracle["gap"]),
])], ignore_index=True).to_csv(f"{D1}/control_validation.csv", index=False)
print(f"control_validation.csv  (best config = {best_name} at {best:.3f}; random {np.mean(rand):.3f})")

# ------------------------------------------------------------------ D1/onoff_*.csv
for name, _ in CONFIGS:
    p = P[name]; on, off = [], []
    for c in CELLS:
        m = np.flatnonzero(cl == c)
        S = sorted({g for i in m for g in np.argsort(-np.abs(TRUE[i]))[:50]})
        for i in m:
            on.append(np.corrcoef(p[i, S], TRUE[i, S])[0, 1])
            off += [np.corrcoef(p[i, S], TRUE[j, S])[0, 1] for j in m if j != i]
    pd.DataFrame({"score": on + off, "kind": ["on"] * len(on) + ["off"] * len(off)}).to_csv(
        f"{D1}/onoff_{name.replace(' ', '_')}.csv", index=False)
print("onoff_*.csv x7")

# ------------------------------------------------------------------ inter-drug matrices
loss = P["loss-only"]
for c in CELLS:
    m = np.flatnonzero(cl == c)
    np.save(f"{D2}/interdrug_{c}.npy", np.corrcoef(loss[m]))
a = np.flatnonzero(cl == "A549")
np.save(f"{D1}/interdrug_corr_lossonly.npy", np.corrcoef(loss[a]))
C = np.corrcoef(loss[a]); iu = np.triu_indices_from(C, 1)
json.dump({"cell": "A549", "drugs": [SHORT.get(d, d) for d in dr[a]],
           "mean_offdiag": float(C[iu].mean())}, open(f"{D1}/interdrug_meta.json", "w"))
print(f"interdrug matrices  (A549 mean off-diagonal = {C[iu].mean():.4f})")

# ------------------------------------------------------------------ D2/per_cellline.csv
pc = []
for name, _ in CONFIGS:
    for c in CELLS:
        m = np.flatnonzero(cl == c)
        sub = dds(P[name][m], TRUE[m], cl[m], top_k=50, metric="pearson")
        Cc = np.corrcoef(P[name][m]); iu2 = np.triu_indices_from(Cc, 1)
        pc.append(dict(config=name, cell=c, auc=sub["specificity_auc"], inter_drug_r=float(Cc[iu2].mean())))
pd.DataFrame(pc).to_csv(f"{D2}/per_cellline.csv", index=False)

# ------------------------------------------------------------------ D2 calibration
cal = [dict(alpha=a_, auc=dds((1 - a_) * mean_pred + a_ * TRUE, TRUE, cl, top_k=50,
                              metric="pearson")["specificity_auc"])
       for a_ in np.round(np.arange(0, 1.01, 0.05), 3)]
pd.DataFrame(cal).to_csv(f"{D2}/calibration.csv", index=False)
grid = np.round(np.arange(0, 0.2001, 0.005), 4)
fine = [dds((1 - a_) * mean_pred + a_ * TRUE, TRUE, cl, top_k=50, metric="pearson")["specificity_auc"]
        for a_ in grid]
eff = float(grid[int(np.argmin(np.abs(np.array(fine) - best)))])
json.dump({"cytobridge_auc": best, "effective_alpha": eff}, open(f"{D2}/calibration_meta.json", "w"))
print(f"calibration  (effective alpha = {eff})")

# ------------------------------------------------------------------ D2 bootstrap over drugs
uniq = sorted(set(dr))
boot = []
for _ in range(1000):
    take = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.flatnonzero(dr == d) for d in take])
    try:
        boot.append(dds(P['loss-only'][idx], TRUE[idx], cl[idx], top_k=50, metric="pearson")["specificity_auc"])
    except Exception:
        continue
boot = np.array(boot)
np.save(f"{D2}/bootstrap_auc.npy", boot)
json.dump({"auc_mean": float(boot.mean()), "auc_lo": float(np.percentile(boot, 2.5)),
           "auc_hi": float(np.percentile(boot, 97.5))}, open(f"{D2}/bootstrap_meta.json", "w"))
print(f"bootstrap  mean {boot.mean():.4f}  95% CI [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")

# ------------------------------------------------------------------ D2 gene variance
np.savez(f"{D2}/gene_variance.npz",
         pred_std=np.stack([loss[cl == c] for c in CELLS]).std(1).mean(0),
         true_std=np.stack([TRUE[cl == c] for c in CELLS]).std(1).mean(0))

# ------------------------------------------------------------------ D2 confusion (A549)
S = sorted({g for i in a for g in np.argsort(-np.abs(TRUE[i]))[:50]})
M = np.array([[np.corrcoef(loss[i, S], TRUE[j, S])[0, 1] for j in a] for i in a])
np.save(f"{D2}/confusion_A549.npy", M)
json.dump({"cell": "A549", "n": len(a), "chance": 1 / len(a),
           "top1_recovery": float(np.mean(M.argmax(1) == np.arange(len(a))))},
          open(f"{D2}/confusion_meta.json", "w"))
print(f"confusion  top-1 = {np.mean(M.argmax(1)==np.arange(len(a))):.4f}  (chance {1/len(a):.4f})")

# ------------------------------------------------------------------ case studies: 4 A549 pairs
import itertools
pairs = sorted(((float(np.corrcoef(loss[i], loss[j])[0, 1]), float(np.corrcoef(TRUE[i], TRUE[j])[0, 1]),
                 SHORT.get(dr[i], dr[i]), SHORT.get(dr[j], dr[j]))
                for i, j in itertools.combinations(a, 2)), key=lambda x: x[1])
chosen = [pairs[0], pairs[1], pairs[len(pairs) // 2], pairs[-1]]
cs = {"cell": "A549"}
for k, (pr_, tr_, A, B) in enumerate(chosen):
    cs[f"pair{k}"] = json.dumps({"A": A, "B": B, "pred_r": pr_, "true_r": tr_})
json.dump(cs, open(f"{D2}/casestudies.json", "w"))
i0, j0 = [x for x in itertools.combinations(a, 2)
          if {SHORT.get(dr[x[0]], dr[x[0]]), SHORT.get(dr[x[1]], dr[x[1]])} == {chosen[0][2], chosen[0][3]}][0]
np.savez(f"{D1}/casestudy.npz", cell="A549", drugA=chosen[0][2], drugB=chosen[0][3],
         predA=loss[i0], trueA=TRUE[i0], predB=loss[j0], trueB=TRUE[j0],
         pred_corr=chosen[0][0], true_corr=chosen[0][1])
print("case studies:")
for pr_, tr_, A, B in chosen:
    print(f"   pred r={pr_:.3f}  true r={tr_:+.3f}   {A} / {B}")
print("\nall figure-backing data regenerated")
