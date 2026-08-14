"""
compute_chemcpa_control.py - the LEARNED baseline arm of the drug-discrimination
control panel. chemCPA (theislab/chemCPA, Hetzel et al. 2022) is the purpose-built
drug-conditioned model a reviewer will expect; this scores whether it discriminates
held-out drugs on the same 9-drug sci-Plex holdout where Ridge (linear) and
CytoBridge (cross-modal) both fail.

chemCPA prediction vectors are produced on cityu (env kagpert-cpa-baseline,
repo ~/chemcpa_build/chemCPA) and copied to
  manuscript/analysis/chemcpa_vectors/{pred_treated_pb,true_treated_pb,control_pb,meta}.*

Like Ridge, chemCPA's reconstruction lives in its OWN consistent space (its dose
pooling + vehicle control), which need NOT match E6E7's canonical reconstruction
(established: the two reconstructions correlate only ~0.26 because the original
dose/control provenance was lost). So chemCPA is scored against ITS OWN true, the
same self-consistent rule used for Ridge. The metric is predictor-intrinsic: given
(pred, true) from one pipeline, does the predictor's per-drug pattern match the
right drug better than wrong drugs? Oracle = 1.0 and Mean ~= 0.50 in ANY consistent
space, which is the invariant this script enforces before trusting the chemCPA AUC.

Reuses the student's EXACT control function; no metric reimplementation (铁律 1:
supervisor computes the reported number, codex only produced the vectors).
Outputs to manuscript/analysis/data_chemcpa/.
"""
import os, sys, json
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score

CV = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/chemcpa_vectors"
BV = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/baseline_vectors_sub"
OUT = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis/data_chemcpa"
os.makedirs(OUT, exist_ok=True)

# canonical (drug, cell_line) order = the 9 holdout drugs x {A549,K562,MCF7}
DRUGS9 = ["AG-490 (Tyrphostin B42)", "Celecoxib", "Fulvestrant", "Ramelteon",
          "SL-327", "SRT3025 HCl", "Thalidomide",
          "Tofacitinib (CP-690550) Citrate", "Zileuton"]
CELLS = ["A549", "K562", "MCF7"]
CANON_KEYS = [f"{d}||{c}" for d in DRUGS9 for c in CELLS]


def control_panel(pred, true, cl, label):
    """Student's exact function: top-50 DEG pearson (primary) + spearman-all
    sensitivity point + inter-drug pearson of the PREDICTIONS (collapse signature)."""
    r50 = drug_discrimination_score(pred, true, cl, top_k=50, metric="pearson")
    rall = drug_discrimination_score(pred, true, cl, top_k=None, metric="spearman")
    inter = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                inter.append(np.corrcoef(pred[m[i]], pred[m[j]])[0, 1])
    return {"predictor": label, "auc_deg50": r50["specificity_auc"],
            "gap_deg50": r50["gap"], "on_diag_deg50": r50["on_diag_mean"],
            "wilcoxon_p_on_gt_off": r50["wilcoxon_p_on_gt_off"],
            "auc_all_spearman": rall["specificity_auc"],
            "inter_drug_pearson": float(np.mean(inter)) if inter else np.nan}


def reindex(mat, meta):
    """Reorder rows to the canonical 9x3 order by (drug||cell_line) key."""
    key = (meta["drug"].astype(str) + "||" + meta["cell_line"].astype(str)).values
    pos = {k: i for i, k in enumerate(key)}
    missing = [k for k in CANON_KEYS if k not in pos]
    if missing:
        raise SystemExit(f"chemcpa meta missing {len(missing)} canonical pairs, "
                         f"e.g. {missing[:3]}")
    return mat[[pos[k] for k in CANON_KEYS]]


def expand_control(ctrl, meta_order):
    """control_pb may be (3,3000) per cell line or (27,3000) per pair. Return (27,3000)
    in the canonical row order."""
    cl = [k.split("||")[1] for k in CANON_KEYS]
    if ctrl.shape[0] == 27:
        return ctrl  # assumed already reindexed by caller
    if ctrl.shape[0] == 3:
        # rows are A549,K562,MCF7 in that order per the spec
        cidx = {c: i for i, c in enumerate(CELLS)}
        return np.stack([ctrl[cidx[c]] for c in cl])
    raise SystemExit(f"unexpected control_pb shape {ctrl.shape}")


if not os.path.exists(f"{CV}/pred_treated_pb.npy"):
    raise SystemExit(f"chemCPA vectors not present yet at {CV} (waiting on cityu run).")

meta = pd.read_csv(f"{CV}/meta.csv")
pred_t = reindex(np.load(f"{CV}/pred_treated_pb.npy"), meta)
true_t = reindex(np.load(f"{CV}/true_treated_pb.npy"), meta)
ctrl_raw = np.load(f"{CV}/control_pb.npy")
# control: if (27,3000) reindex by meta, else expand from (3,3000)
ctrl = reindex(ctrl_raw, meta) if ctrl_raw.shape[0] == len(meta) else expand_control(ctrl_raw, None)

cl_canon = np.array([k.split("||")[1] for k in CANON_KEYS])

# chemCPA's X is already log-normalized (range ~0..6), so the pseudobulks are means
# of log-normalized expression. logFC is therefore a plain difference treated-control,
# NOT log1p again. Guard: if the pseudobulks ever look like raw counts (max>30), fall
# back to log1p so the script stays correct under either convention.
if max(float(pred_t.max()), float(true_t.max())) > 30:
    pred_logfc = np.log1p(np.clip(pred_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    true_logfc = np.log1p(np.clip(true_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    print("[space] raw-count pseudobulks detected -> logFC = log1p(treated)-log1p(control)")
else:
    pred_logfc = pred_t - ctrl
    true_logfc = true_t - ctrl
    print("[space] log-normalized pseudobulks detected -> logFC = treated - control")
print(f"chemCPA vectors: pred {pred_logfc.shape}, true {true_logfc.shape}, "
      f"control {ctrl.shape}")

# ---- INVARIANT: the chemCPA self-space must be a WELL-POSED scoring space ----
# Oracle (true vs true) must be 1.0; Mean (cell-line avg of true) ~0.50; Random ~0.50.
# If these anchors are off, the reconstruction/units are broken -> do not trust AUC.
oracle = control_panel(true_logfc.copy(), true_logfc, cl_canon, "Oracle (pos ctrl)")
mean_pred = np.zeros_like(true_logfc)
for c in np.unique(cl_canon):
    m = np.flatnonzero(cl_canon == c)
    mean_pred[m] = true_logfc[m].mean(axis=0)
mean_row = control_panel(mean_pred, true_logfc, cl_canon, "Mean (neg ctrl)")
rng = np.random.default_rng(0)
rand_row = control_panel(true_logfc[rng.permutation(len(true_logfc))], true_logfc,
                         cl_canon, "Random (neg ctrl)")
print(f"  [invariant] Oracle AUC={oracle['auc_deg50']:.3f} (must be 1.000), "
      f"Mean AUC={mean_row['auc_deg50']:.3f} (must be ~0.50)")
if oracle["auc_deg50"] < 0.999:
    raise SystemExit("SELF-SPACE INVALID: Oracle != 1.0 -> chemCPA true/units broken.")
if abs(mean_row["auc_deg50"] - 0.5) > 0.08:
    print("  [warn] Mean AUC deviates from 0.50 by >0.08 -> inspect cell-line balance.")

# ---- chemCPA control (self-space, comparable to Ridge's self-space) ----
chem_row = control_panel(pred_logfc, true_logfc, cl_canon, "chemCPA (self-space)")

# ---- SECONDARY cross-space check vs the Ridge agent-space sub-true (informational) ----
rows = [rand_row, mean_row, chem_row, oracle]
if os.path.exists(f"{BV}/true.npy"):
    sub_true = np.load(f"{BV}/true.npy")
    sub_meta = pd.read_csv(f"{BV}/meta.csv")
    sub_true = reindex(sub_true, sub_meta)
    xcorr = np.corrcoef(true_logfc.ravel(), sub_true.ravel())[0, 1]
    cross = control_panel(pred_logfc, sub_true, cl_canon,
                          "chemCPA vs Ridge-space true (cross, informational)")
    cross["note"] = f"true-space corr chemCPA-vs-Ridge={xcorr:.3f}"
    rows.append(cross)
    print(f"  [secondary] chemCPA-space true vs Ridge-space true corr={xcorr:.3f}; "
          f"cross-space AUC={cross['auc_deg50']:.3f} (informational only)")

panel = pd.DataFrame(rows)
panel.to_csv(f"{OUT}/chemcpa_control_panel.csv", index=False)
np.save(f"{OUT}/chemcpa_pred_logfc.npy", pred_logfc.astype(np.float32))
np.save(f"{OUT}/chemcpa_true_logfc.npy", true_logfc.astype(np.float32))
with open(f"{OUT}/chemcpa_summary.json", "w") as fh:
    json.dump({"chemcpa": chem_row, "oracle": oracle, "mean": mean_row,
               "random": rand_row}, fh, indent=2, default=float)

print("\n=== chemCPA drug-discrimination control (self-space) ===")
print(panel.round(3).to_string(index=False))
print(f"\nHEADLINE: chemCPA AUC@50(DEG,pearson) = {chem_row['auc_deg50']:.3f}, "
      f"inter-drug pearson = {chem_row['inter_drug_pearson']:.3f}")
print(f"[done] -> {OUT}")
