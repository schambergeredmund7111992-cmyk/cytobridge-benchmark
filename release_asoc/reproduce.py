#!/usr/bin/env python3
"""Regenerate every number reported in the manuscript, from the artifacts in this release.

    python3 reproduce.py

No training, no GPU, no network. Reads the frozen split and the stored predictions,
recomputes each reported quantity with the paper's own metric implementation
(code/metrics.py, unmodified), and prints PAPER vs RECOMPUTED with a verdict per row.

Requires: numpy, pandas, scipy  (pyarrow or fastparquet to read the split parquet).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
from metrics import drug_discrimination_score as dds          # noqa: E402
from metrics import inter_drug_pearson                         # noqa: E402

TOL = 0.0006
fails = []


def check(label, paper, got, tol=TOL, fmt="{:.3f}"):
    ok = abs(paper - got) <= tol
    if not ok:
        fails.append(label)
    print(f"  {label:<44s} paper {fmt.format(paper):>8s}   recomputed {fmt.format(got):>8s}   "
          f"{'PASS' if ok else 'FAIL'}")


def head(t):
    print(f"\n{'='*84}\n{t}\n{'='*84}")


# ---------------------------------------------------------------- 0. split integrity
head("0.  Frozen split — the only split used anywhere in the paper")
sp = json.load(open(f"{ROOT}/split/internal_splits.json"))
tr, va, te = (set(sp[k]) for k in ("train_drugs", "val_drugs", "test_drugs"))
print(f"  train {len(tr)}   val {len(va)}   test {len(te)}")
for a, b, n in ((tr, te, "train ∩ test"), (va, te, "val ∩ test"), (tr, va, "train ∩ val")):
    ok = not (a & b)
    print(f"  {n:<44s} {'empty  PASS' if ok else f'{sorted(a & b)}  FAIL'}")
    if not ok:
        fails.append(n)
print(f"  held-out compounds: {', '.join(sorted(te))}")

# ---------------------------------------------------------------- rebuild pair table
man = pd.read_parquet(f"{ROOT}/split/sciplex_test.parquet")
tre = np.load(f"{ROOT}/split/sciplex_test_treated_counts.npy", mmap_mode="r")
ctl = np.load(f"{ROOT}/split/sciplex_test_control_counts.npy", mmap_mode="r")
drug = man["drug_id"].astype(str).to_numpy()
cell = man["cell_line"].astype(str).to_numpy()

keys, cp, tp = [], [], []
for (d, c), g in pd.DataFrame({"drug": drug, "cell": cell}).groupby(["drug", "cell"]):
    i = np.asarray(list(g.index), dtype=int)
    keys.append((d, c))
    cp.append(np.asarray(ctl[i]).mean(0))
    tp.append(np.asarray(tre[i]).mean(0))
cp, tp = np.stack(cp), np.stack(tp)
cl = np.array([c for _, c in keys])
cline = np.stack([np.asarray(ctl[cell == c]).mean(0) for c in cl])   # one vehicle per cell line
SHIFT = np.log1p(cp) - np.log1p(cline)
TRUE = np.log1p(tp) - np.log1p(cline)

STEMS = [("loss-only", "t7_sub_loss_only"), ("drug-spec x1", "t7_sub_drugspec1"),
         ("norm-only", "t7_sub_norm_only"), ("recovery baseline", "t6_sub_baseline"),
         ("low recon weight", "t7_sub_lamrecon01"), ("drug-spec x5", "t7_sub_drugspec5"),
         ("drug-spec x3", "t7_sub_drugspec3")]


def load(stem):
    pr = np.load(f"{ROOT}/predictions/cytobridge/logfc_pred_{stem}.npy")
    tu = np.load(f"{ROOT}/predictions/cytobridge/logfc_true_{stem}.npy")
    mt = pd.read_csv(f"{ROOT}/predictions/cytobridge/logfc_meta_{stem}.csv")
    inv = np.argsort([keys.index((r.drug, r.cell_line)) for r in mt.itertuples()])
    err = float(np.abs((np.log1p(tp) - np.log1p(cp)) - tu[inv]).max())
    return pr[inv] + SHIFT, err


head("1.  Known-answer gate — rebuild the stored targets from the frozen counts")
for name, stem in STEMS:
    _, err = load(stem)
    ok = err < 1e-5
    if not ok:
        fails.append(f"gate {name}")
    print(f"  {name:<44s} max|rebuilt - stored| = {err:.2e}   {'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------- 2. Table: collapse
head("2.  Table 4 — the off-diagonal control across seven loss configurations")
PAPER = {"norm-only": (0.981, 0.519, 0.038, 0.214), "recovery baseline": (0.981, 0.523, 0.029, 0.250),
         "drug-spec x1": (0.972, 0.523, 0.021, 0.207), "low recon weight": (0.983, 0.542, 0.020, 0.250),
         "loss-only": (0.981, 0.509, 0.014, 0.228), "drug-spec x5": (0.986, 0.532, 0.010, 0.314),
         "drug-spec x3": (0.988, 0.500, 0.001, 0.458)}
print(f"  {'configuration':<22s}{'r_inter':>18s}{'AUC':>18s}{'gap':>13s}{'p':>13s}")
aucs = {}
for name, stem in STEMS:
    P, _ = load(stem)
    r = dds(P, TRUE, cl, top_k=50, metric="pearson")
    got = (inter_drug_pearson(P, cl), r["specificity_auc"], r["gap"], r["wilcoxon_p_on_gt_off"])
    aucs[name] = got[1]
    pa = PAPER[name]
    bad = [i for i in range(4) if abs(pa[i] - got[i]) > 0.0011]
    if bad:
        fails.append(f"tab4 {name}")
    print(f"  {name:<22s}" + "".join(
        f"{pa[i]:.3f}/{got[i]:.3f}".rjust(18 if i < 2 else 13) for i in range(4))
        + ("   PASS" if not bad else "   FAIL"))

# ---------------------------------------------------------------- 3. the missing anchor
head("3.  The control's own anchor — a predictor given no drug information")
blind = np.log1p(np.tile(tp.mean(0), (len(keys), 1)))
a_pair = dds(blind - np.log1p(cp), np.log1p(tp) - np.log1p(cp), cl, top_k=50, metric="pearson")
a_line = dds(blind - np.log1p(cline), TRUE, cl, top_k=50, metric="pearson")
check("no-drug-info, per-pair vehicle (AUC)", 0.588, a_pair["specificity_auc"])
check("no-drug-info, per-pair vehicle (gap)", 0.092, a_pair["gap"])
check("no-drug-info, pooled vehicle  (AUC)", 0.500, a_line["specificity_auc"])
best = max(aucs.values())
print(f"\n  best configuration = {best:.3f};  no-drug-information predictor under the per-pair\n"
      f"  vehicle = {a_pair['specificity_auc']:.3f}  ->  the anchor outscores every model "
      f"{'(as reported)' if a_pair['specificity_auc'] > best else '(DISCREPANCY)'}")

# ---------------------------------------------------------------- 4. well-posedness
head("4.  Well-posedness of the endpoint (must hold before any verdict is read)")
mean_pred = np.zeros_like(TRUE)
for c in np.unique(cl):
    mean_pred[cl == c] = TRUE[cl == c].mean(0)
check("Oracle  (own measured response)", 1.000, dds(TRUE, TRUE, cl, top_k=50, metric="pearson")["specificity_auc"])
check("Mean    (cell-line average)", 0.500, dds(mean_pred, TRUE, cl, top_k=50, metric="pearson")["specificity_auc"])

# ---------------------------------------------------------------- 5. recovered fraction
head("5.  Recovered fraction against the measured biological ceiling")
CEIL = 0.810
phi = (best - 0.5) / (CEIL - 0.5) * 100
check("phi, best configuration (%)", 13.0, phi, tol=0.6, fmt="{:.1f}")
print(f"  ceiling {CEIL} is measured from two sci-Plex replicates on disjoint plates;\n"
      f"  see analysis/supp_T8/replicate_reliability_27.py in the manuscript repository.")

# ---------------------------------------------------------------- verdict
head("VERDICT")
if fails:
    print(f"  {len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("  All reported numbers reproduce from the released artifacts.")
