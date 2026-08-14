"""
manuscript/analysis/supp_T6/metric_degk_grid.py
===============================================
T6 (Xichen task book, section T6) — metric x DEG-k sensitivity grid.

WHY: a reviewer can object that the "collapse" verdict (control AUC ~= 0.5) is an
artifact of one metric / one DEG-k choice. This sweeps DEG-k in {20, 50, 100, all}
x metric in {pearson, spearman} and shows the verdict is stable across the grid:
the model predictor never discriminates drugs, while Oracle stays 1.0 and Mean ~= 0.5
at EVERY grid cell (the well-posedness gate, asserted across the whole grid).

CONSTRUCTION NOTE: on the per-pair-vehicle grid four cells (all-3000 x Spearman)
exceed the 0.70 threshold; the paper discusses this exception explicitly
(Section 4.5) and reports the pooled-vehicle grid in Table 7, where all 56 cells
fall in [0.481, 0.574]. See manuscript/analysis/data3/README.md.

REUSE, DO NOT REWRITE: scores with code/eval/metrics.py::drug_discrimination_score.

NOTE ON COSINE (task book also lists cosine): drug_discrimination_score only supports
"pearson"/"spearman". Adding a cosine branch would modify the SHARED reported-number
metric -> that is a supervisor decision (铁律 1), NOT done here. This grid is
pearson x spearman; cosine is left as a flagged extension.

REPORTED-NUMBER OWNERSHIP: supervisor-authored. `--selftest` is the local proof
(numpy/scipy only, no external data).

USAGE
-----
  python metric_degk_grid.py --selftest
  # real run on an existing CytoBridge config artifact (default t7_sub_loss_only):
  python metric_degk_grid.py --artifact t7_sub_loss_only --out_dir .
  # or point at explicit vectors:
  python metric_degk_grid.py --pred P.npy --true T.npy --meta M.csv --out_dir .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CODE = _REPO / "code"
_DEFAULT_RES = _REPO / "student_progress_E6E7" / "E6E7" / "results"

KS = [20, 50, 100, None]          # None = all genes
METRICS = ["pearson", "spearman"]


def _import_metric(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(f"--code_dir {code_dir} has no eval/metrics.py; point it at code/.")
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def mean_pred(true, cl):
    out = np.zeros_like(true)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        out[m] = true[m].mean(axis=0)
    return out


def build_grid(score_fn, pred, true, cl):
    rows = []
    for k in KS:
        for metric in METRICS:
            klabel = "all" if k is None else k
            cell = {"deg_k": klabel, "metric": metric}
            for label, P in [("Model", pred), ("Mean", mean_pred(true, cl)),
                             ("Oracle", true.copy())]:
                r = score_fn(P, true, cl, top_k=k, metric=metric)
                cell[f"auc_{label}"] = round(float(r["specificity_auc"]), 4)
                if label == "Model":
                    cell["gap_Model"] = round(float(r["gap"]), 4)
                    cell["wilcoxon_p_Model"] = round(float(r["wilcoxon_p_on_gt_off"]), 4)
            rows.append(cell)
    return pd.DataFrame(rows)


def assert_well_posed_grid(grid: pd.DataFrame):
    bad_o = grid[grid["auc_Oracle"] < 0.999]
    bad_m = grid[(grid["auc_Mean"] - 0.5).abs() > 0.06]
    print(f"[gate] Oracle==1.0 at all {len(grid)} cells: {bad_o.empty}; "
          f"Mean~=0.5 at all cells: {bad_m.empty}")
    if not bad_o.empty:
        raise SystemExit(f"WELL-POSEDNESS FAIL: Oracle != 1.0 at\n{bad_o}")
    if not bad_m.empty:
        raise SystemExit(f"WELL-POSEDNESS FAIL: Mean != ~0.5 at\n{bad_m}")


def load_artifact(args):
    if args.pred and args.true and args.meta:
        pred, true = np.load(args.pred), np.load(args.true)
        meta = pd.read_csv(args.meta)
    else:
        res = Path(args.res_dir)
        pred = np.load(res / f"logfc_pred_{args.artifact}.npy")
        true = np.load(res / f"logfc_true_{args.artifact}.npy")
        meta = pd.read_csv(res / f"logfc_meta_{args.artifact}.csv")
    cl = meta["cell_line"].astype(str).values
    return np.asarray(pred, float), np.asarray(true, float), cl


def run_real(args):
    score_fn = _import_metric(args.code_dir)
    pred, true, cl = load_artifact(args)
    print(f"[load] pred {pred.shape}, true {true.shape}, {len(np.unique(cl))} cell lines")
    grid = build_grid(score_fn, pred, true, cl)
    assert_well_posed_grid(grid)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "metric_degk_grid.csv", index=False)
    print("\n=== metric x DEG-k sensitivity grid (Model vs Mean vs Oracle) ===")
    print(grid.to_string(index=False))
    print(f"\n[done] -> {out}/metric_degk_grid.csv")
    print("Interpret: if auc_Model stays ~0.5 across the WHOLE grid, the collapse "
          "verdict does not depend on the metric/DEG-k choice.")


def run_selftest(args):
    score_fn = _import_metric(args.code_dir)
    rng = np.random.default_rng(11)
    n_cl, n_drug, n_gene = 3, 6, 300
    drug_sig = rng.normal(0, 1.0, (n_drug, n_gene))
    cell_off = rng.normal(0, 0.5, (n_cl, n_gene))
    true, signal, cl = [], [], []
    for c in range(n_cl):
        for d in range(n_drug):
            true.append(drug_sig[d] + cell_off[c] + rng.normal(0, 0.1, n_gene))
            signal.append(drug_sig[d] + cell_off[c] + rng.normal(0, 0.5, n_gene))
            cl.append(f"CL{c}")
    true, signal, cl = np.array(true), np.array(signal), np.array(cl)

    grid = build_grid(score_fn, signal, true, cl)  # 'Model' row = planted signal
    print("\n=== SELFTEST grid (Model col = planted signal) ===")
    print(grid.to_string(index=False))
    assert_well_posed_grid(grid)
    sig_ok = bool((grid["auc_Model"] > 0.7).all())
    print(f"\n[selftest] Signal auc>0.7 at every cell: {sig_ok}")
    if not sig_ok:
        raise SystemExit("SELFTEST FAILED: planted signal not discriminated across grid.")
    print("SELFTEST PASSED: grid scoring + well-posedness gate verified across "
          f"{len(grid)} cells (k x metric).")


def main():
    p = argparse.ArgumentParser(description="T6 metric x DEG-k sensitivity grid")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--artifact", default="t7_sub_loss_only",
                   help="E6E7 config name -> logfc_{pred,true,meta}_<artifact>.{npy,csv}")
    p.add_argument("--res_dir", type=Path, default=_DEFAULT_RES)
    p.add_argument("--pred", type=Path); p.add_argument("--true", type=Path)
    p.add_argument("--meta", type=Path)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--code_dir", type=Path, default=_DEFAULT_CODE)
    args = p.parse_args()
    if args.selftest:
        run_selftest(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
