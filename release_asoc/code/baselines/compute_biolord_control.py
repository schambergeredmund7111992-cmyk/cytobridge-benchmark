"""
compute_biolord_control.py - the biolord arm of the drug-discrimination control panel.

biolord (nitzanlab/biolord, Wagner/Nitzan Nat Biotech 2024) is a disentanglement model
PURPOSE-BUILT for unseen-drug sci-Plex prediction (its paper predicts 9 unseen sci-Plex
drugs via a molecular drug embedding, evaluated by r2). It is therefore the strongest
external SOTA to audit: if a model designed for unseen-drug sci-Plex ALSO fails the
off-diagonal control where Ridge (linear), chemCPA (learned) and CytoBridge (cross-modal)
already fail, the task-level claim is maximally supported.

biolord prediction vectors are produced on cityu (env `biolord`, repo biolord_reproducibility
sci-Plex notebook adapted to OUR 9-drug holdout) and copied to
  manuscript/analysis/biolord_vectors/{pred_treated_pb,true_treated_pb,control_pb,meta}.*

SELF-SPACE SCORING (identical rule to Ridge/chemCPA): biolord is scored against ITS OWN
reconstruction (its dose pooling + vehicle control), which need NOT match E6E7's canonical
reconstruction. The control is predictor-intrinsic; Oracle=1.0 and Mean~=0.50 in ANY
consistent space, the invariant enforced before the biolord AUC is trusted.

Reuses the EXACT student control function (铁律 1: supervisor computes the reported number,
the cityu job only produces the vectors). Outputs to manuscript/analysis/data_biolord/.

Usage:
  python compute_biolord_control.py                 # real run: biolord_vectors/
  python compute_biolord_control.py chemcpa_vectors  # pipeline self-test on existing vectors
"""
import os, sys, json
import numpy as np
import pandas as pd

CODE = "/Users/cgxmac/Desktop/CytoBridge/student_progress_E6E7/E6E7"
sys.path.insert(0, CODE)
from eval.metrics import drug_discrimination_score

HERE = "/Users/cgxmac/Desktop/CytoBridge/manuscript/analysis"
VEC_NAME = sys.argv[1] if len(sys.argv) > 1 else "biolord_vectors"
CV = f"{HERE}/{VEC_NAME}"
BV = f"{HERE}/baseline_vectors_sub"
OUT = f"{HERE}/data_biolord"
os.makedirs(OUT, exist_ok=True)

DRUGS9 = ["AG-490 (Tyrphostin B42)", "Celecoxib", "Fulvestrant", "Ramelteon",
          "SL-327", "SRT3025 HCl", "Thalidomide",
          "Tofacitinib (CP-690550) Citrate", "Zileuton"]
CELLS = ["A549", "K562", "MCF7"]
CANON_KEYS = [f"{d}||{c}" for d in DRUGS9 for c in CELLS]


def control_panel(pred, true, cl, label):
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
    key = (meta["drug"].astype(str) + "||" + meta["cell_line"].astype(str)).values
    pos = {k: i for i, k in enumerate(key)}
    missing = [k for k in CANON_KEYS if k not in pos]
    if missing:
        raise SystemExit(f"biolord meta missing {len(missing)} canonical pairs, e.g. {missing[:3]}")
    return mat[[pos[k] for k in CANON_KEYS]]


def expand_control(ctrl):
    cl = [k.split("||")[1] for k in CANON_KEYS]
    if ctrl.shape[0] == 27:
        return ctrl
    if ctrl.shape[0] == 3:
        cidx = {c: i for i, c in enumerate(CELLS)}
        return np.stack([ctrl[cidx[c]] for c in cl])
    raise SystemExit(f"unexpected control_pb shape {ctrl.shape}")


if not os.path.exists(f"{CV}/pred_treated_pb.npy"):
    raise SystemExit(f"biolord vectors not present yet at {CV} (waiting on cityu run).")

meta = pd.read_csv(f"{CV}/meta.csv")
pred_t = reindex(np.load(f"{CV}/pred_treated_pb.npy"), meta)
true_t = reindex(np.load(f"{CV}/true_treated_pb.npy"), meta)
ctrl_raw = np.load(f"{CV}/control_pb.npy")
ctrl = reindex(ctrl_raw, meta) if ctrl_raw.shape[0] == len(meta) else expand_control(ctrl_raw)

cl_canon = np.array([k.split("||")[1] for k in CANON_KEYS])

# adaptive space: raw-count pseudobulks -> log1p diff; log-normalized -> plain diff.
if max(float(pred_t.max()), float(true_t.max())) > 30:
    pred_logfc = np.log1p(np.clip(pred_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    true_logfc = np.log1p(np.clip(true_t, 0, None)) - np.log1p(np.clip(ctrl, 0, None))
    print("[space] raw-count pseudobulks -> logFC = log1p(treated)-log1p(control)")
else:
    pred_logfc = pred_t - ctrl
    true_logfc = true_t - ctrl
    print("[space] log-normalized pseudobulks -> logFC = treated - control")
print(f"biolord vectors: pred {pred_logfc.shape}, true {true_logfc.shape}, control {ctrl.shape}")

# ---- INVARIANT: self-space must be well-posed (Oracle=1.0, Mean~0.5, Random~0.5) ----
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
    raise SystemExit("SELF-SPACE INVALID: Oracle != 1.0 -> biolord true/units broken.")
if abs(mean_row["auc_deg50"] - 0.5) > 0.08:
    print("  [warn] Mean AUC deviates from 0.50 by >0.08 -> inspect cell-line balance.")

# ---- biolord control (self-space, comparable to Ridge/chemCPA self-space) ----
bio_row = control_panel(pred_logfc, true_logfc, cl_canon, "biolord (self-space)")

# ---- SECONDARY cross-space check vs the Ridge agent-space sub-true (informational) ----
rows = [rand_row, mean_row, bio_row, oracle]
if os.path.exists(f"{BV}/true.npy"):
    sub_true = reindex(np.load(f"{BV}/true.npy"), pd.read_csv(f"{BV}/meta.csv"))
    xcorr = np.corrcoef(true_logfc.ravel(), sub_true.ravel())[0, 1]
    cross = control_panel(pred_logfc, sub_true, cl_canon,
                          "biolord vs Ridge-space true (cross, informational)")
    cross["note"] = f"true-space corr biolord-vs-Ridge={xcorr:.3f}"
    rows.append(cross)
    print(f"  [secondary] biolord-space true vs Ridge-space true corr={xcorr:.3f}; "
          f"cross-space AUC={cross['auc_deg50']:.3f} (informational only)")

panel = pd.DataFrame(rows)
panel.to_csv(f"{OUT}/biolord_control_panel.csv", index=False)
np.save(f"{OUT}/biolord_pred_logfc.npy", pred_logfc.astype(np.float32))
np.save(f"{OUT}/biolord_true_logfc.npy", true_logfc.astype(np.float32))
with open(f"{OUT}/biolord_summary.json", "w") as fh:
    json.dump({"biolord": bio_row, "oracle": oracle, "mean": mean_row, "random": rand_row},
              fh, indent=2, default=float)

print("\n=== biolord drug-discrimination control (self-space) ===")
print(panel.round(3).to_string(index=False))
print(f"\nHEADLINE: biolord AUC@50(DEG,pearson) = {bio_row['auc_deg50']:.3f}, "
      f"inter-drug pearson = {bio_row['inter_drug_pearson']:.3f}")
print(f"[done] -> {OUT}")
