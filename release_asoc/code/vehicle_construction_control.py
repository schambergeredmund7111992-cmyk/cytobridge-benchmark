"""Re-score every CytoBridge config with a per-cell-line vehicle instead of a per-pair one.

The paper's CytoBridge numbers came from eval/save_predictions.py::aggregate, which does

    ctrl_pb = ctrl_counts[idx].mean(0)              # idx = THIS pair's rows -> per-pair
    pred_i  = log1p(mu_pb)      - log1p(ctrl_pb)
    true_i  = log1p(treated_pb) - log1p(ctrl_pb)    # SAME ctrl_pb

so a term that is absent from every off-diagonal target sits in both arguments of the
on-diagonal similarity. Every other model row in the paper (Mean/Ridge via
produce_baseline_control_clean9.py:169-171, chemCPA/biolord via their stored control_pb.npy,
and the replicate ceiling via replicate_reliability_27.py:84-102) uses ONE vehicle per cell
line, which cancels on both diagonals.

This script rebuilds ctrl_pb from the frozen splits_sub arrays, inverts the stored logFC back
to log1p(mu_pb), and re-scores against a per-cell-line vehicle. Nothing is re-trained and no
model is re-run: the inversion is exact and is gated on a known-answer check against the
stored truth arrays.
"""
import sys
import numpy as np
import pandas as pd

E6E7 = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, E6E7)
from eval.metrics import drug_discrimination_score as dds  # the paper's own metric

SPL = ("/Users/cgxmac/Desktop/CytoBridge/student_progress_CytoBridge_收尾交接_赵希宸/汇报/成果/X1_canonical_split/splits_canonical")

RES = f"{E6E7}/results"
CONFIGS = ["loss_only", "norm_only", "lamrecon01", "drugspec1", "drugspec3", "drugspec5"]  # + t6 recovery baseline handled below
TOL = 1e-5

# ---------------------------------------------------------------- frozen split -> pair table
man = pd.read_parquet(f"{SPL}/sciplex_test.parquet")
tre = np.load(f"{SPL}/sciplex_test_treated_counts.npy", mmap_mode="r")
ctl = np.load(f"{SPL}/sciplex_test_control_counts.npy", mmap_mode="r")
print(f"manifest {man.shape}  treated {tre.shape}  control {ctl.shape}")
assert len(man) == tre.shape[0] == ctl.shape[0], "row counts disagree"

drug = man["drug_id"].astype(str).to_numpy()
cell = man["cell_line"].astype(str).to_numpy()

# Reproduce save_predictions.py::aggregate grouping exactly: groupby(['drug','cell']) is sorted.
keys, ctrl_pair, treat_pair = [], [], []
for (d, c), grp in pd.DataFrame({"drug": drug, "cell": cell}).groupby(["drug", "cell"]):
    idx = np.asarray(list(grp.index), dtype=int)
    keys.append((d, c))
    ctrl_pair.append(np.asarray(ctl[idx]).mean(0))
    treat_pair.append(np.asarray(tre[idx]).mean(0))
ctrl_pair = np.stack(ctrl_pair)
treat_pair = np.stack(treat_pair)
cl = np.array([c for _, c in keys])
print(f"pairs rebuilt: {len(keys)}  cell lines: {sorted(set(cl))}")

# Per-cell-line shared vehicle: pool every vehicle cell of that line (the leak-free construction).
ctrl_line = np.stack([np.asarray(ctl[cell == c]).mean(0) for c in cl])
n_distinct = len(np.unique(ctrl_line, axis=0))
print(f"per-pair vehicles: {len(np.unique(ctrl_pair, axis=0))} distinct | "
      f"per-cell-line vehicles: {n_distinct} distinct")
assert n_distinct == len(set(cl)), "shared vehicle must have one distinct row per cell line"


def score(pred, true):
    r = dds(pred, true, cl, top_k=50, metric="pearson")
    return r["specificity_auc"], r["gap"], r.get("wilcoxon_p_on_gt_off", float("nan"))


print("\n" + "=" * 96)
print("KNOWN-ANSWER GATE: rebuild the stored truth from the frozen counts before trusting anything")
print("=" * 96)
ok_all = True
for cfg in CONFIGS:
    try:
        true_stored = np.load(f"{RES}/logfc_true_t7_sub_{cfg}.npy")
        meta = pd.read_csv(f"{RES}/logfc_meta_t7_sub_{cfg}.csv")
    except FileNotFoundError:
        print(f"  {cfg:12s} SKIP (arrays not present)")
        continue
    order = [keys.index((r.drug, r.cell_line)) for r in meta.itertuples()]
    rebuilt = (np.log1p(treat_pair) - np.log1p(ctrl_pair))[order]
    err = float(np.abs(rebuilt - true_stored).max())
    ok = err < TOL
    ok_all &= ok
    print(f"  {cfg:12s} max|rebuilt-stored| = {err:.3e}   {'PASS' if ok else 'FAIL'}")
if not ok_all:
    raise SystemExit("\nKNOWN-ANSWER GATE FAILED -- inversion is not exact, refusing to report numbers.")
print("  -> inversion is exact; the rescoring below is a pure re-aggregation, not a re-run.")

print("\n" + "=" * 96)
print("CytoBridge, same predictions, two vehicle constructions   (paper's metric, top-50, pearson)")
print("=" * 96)
print(f"{'config':14s} {'per-pair (as published)':>30s} {'per-cell-line (leak-free)':>30s}")
for cfg in CONFIGS:
    try:
        pred_stored = np.load(f"{RES}/logfc_pred_t7_sub_{cfg}.npy")
        true_stored = np.load(f"{RES}/logfc_true_t7_sub_{cfg}.npy")
        meta = pd.read_csv(f"{RES}/logfc_meta_t7_sub_{cfg}.csv")
    except FileNotFoundError:
        continue
    order = [keys.index((r.drug, r.cell_line)) for r in meta.itertuples()]
    inv = np.argsort(order)
    pred_o, true_o = pred_stored[inv], true_stored[inv]     # into `keys` order

    a_old, g_old, p_old = score(pred_o, true_o)
    # invert to counts space, then subtract the shared per-cell-line vehicle instead
    log_mu = pred_o + np.log1p(ctrl_pair)
    log_tr = true_o + np.log1p(ctrl_pair)
    a_new, g_new, p_new = score(log_mu - np.log1p(ctrl_line), log_tr - np.log1p(ctrl_line))
    print(f"{cfg:14s} {a_old:9.4f} (gap {g_old:+.4f}, p {p_old:.3f}) "
          f"{a_new:11.4f} (gap {g_new:+.4f}, p {p_new:.3f})")

print("\n" + "=" * 96)
print("THE MISSING ANCHOR: a predictor that is given NO drug information at all")
print("=" * 96)
blind = np.tile(treat_pair.mean(0), (len(keys), 1))          # one profile for every drug
a1, g1, p1 = score(np.log1p(blind) - np.log1p(ctrl_pair),
                   np.log1p(treat_pair) - np.log1p(ctrl_pair))
a2, g2, p2 = score(np.log1p(blind) - np.log1p(ctrl_line),
                   np.log1p(treat_pair) - np.log1p(ctrl_line))
print(f"  per-pair vehicle      auc = {a1:.4f}  gap = {g1:+.4f}  p = {p1:.2e}   <- truth is 0.5")
print(f"  per-cell-line vehicle auc = {a2:.4f}  gap = {g2:+.4f}  p = {p2:.2e}")
