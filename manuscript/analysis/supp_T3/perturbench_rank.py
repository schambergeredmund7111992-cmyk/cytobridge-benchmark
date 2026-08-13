"""
manuscript/analysis/supp_T3/perturbench_rank.py
===============================================
T3 (Xichen task book, section T3) — PerturBench rank / Perturbation Discrimination Score
on the SAME held-out matrix, beside our off-diagonal control AUC, to show the control is
NOT just a reparameterisation of an existing rank metric.

PerturBench rank metric (arXiv:2408.10609, Eq. 2), implemented faithfully:
    rank(x_hat_i) = (1/(p-1)) * sum_{j != i}  1[ dist(x_hat_j, x_i) <= dist(x_hat_i, x_i) ]
    rank_average  = mean_i rank(x_hat_i)
i.e. the fraction of OTHER predictions at least as close to perturbation i's truth as the
correct prediction. 0 = perfect, 0.5 = random, 1 = worst. The Perturbation Discrimination
Score (PDS, Cell-Eval / STATE) is the same quantity reported as 1 - rank.

We do NOT install PerturBench (its repo pulls a heavy hydra/lightning stack); the metric is
a closed form over our existing (pred, true) arrays. We DO reuse our own
drug_discrimination_score for the side-by-side AUC (not reimplemented).

TWO FRAMINGS (this is the point):
  - GLOBAL: candidate set = all p pairs (PerturBench's native cross-perturbation framing;
    mixes drug- and cell-line structure).
  - WITHIN-CELL-LINE: candidate set restricted to same-cell-line pairs (our control's
    conditioning, which isolates drug-specificity from shared cell-line structure).
The gap between the two framings is the empirical evidence for the methodological
difference; the formal equivalence/argument is written by the supervisor in the paper.

REPORTED-NUMBER OWNERSHIP: supervisor-authored. `--selftest` proves it locally.

USAGE
-----
  python perturbench_rank.py --selftest
  python perturbench_rank.py --artifact t7_sub_loss_only --out_dir .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CODE = _REPO / "code"
_DEFAULT_RES = _REPO / "student_progress_E6E7" / "E6E7" / "results"

DISTANCES = ["cosine", "euclidean"]


def _import_metric(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(f"--code_dir {code_dir} has no eval/metrics.py; point it at code/.")
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def perturbench_rank(pred, true, metric, groups=None):
    """rank_average per PerturBench Eq. 2. If groups is given, the candidate set for each i
    is restricted to indices sharing i's group (within-cell-line framing); otherwise global."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    D = cdist(pred, true, metric=metric)          # D[j, i] = dist(pred_j, true_i)
    p = pred.shape[0]
    groups = np.asarray(groups) if groups is not None else np.zeros(p, int)
    ranks = []
    for i in range(p):
        cand = np.flatnonzero((groups == groups[i]) & (np.arange(p) != i))
        if cand.size == 0:
            continue
        di = D[i, i]                               # dist(correct pred_i, true_i)
        ranks.append(float(np.mean(D[cand, i] <= di)))
    return float(np.mean(ranks)) if ranks else float("nan")


def mean_pred(true, cl):
    out = np.zeros_like(true)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        out[m] = true[m].mean(axis=0)
    return out


def build_table(score_fn, pred, true, cl):
    rng = np.random.default_rng(0)
    rand = true[rng.permutation(len(true))]
    mean_p = mean_pred(true, cl)
    predictors = {"Model": pred, "Mean": mean_p, "Random": rand, "Oracle": true.copy()}
    rows = []
    for framing, groups in [("global", None), ("within_cellline", cl)]:
        for dist in DISTANCES:
            row = {"framing": framing, "distance": dist}
            for name, P in predictors.items():
                row[f"rank_{name}"] = round(perturbench_rank(P, true, dist, groups), 4)
            rows.append(row)
    # our control AUC (within-cell-line, top-50 pearson) for the same predictors, side by side
    auc = {name: round(float(score_fn(P, true, cl, top_k=50, metric="pearson")["specificity_auc"]), 4)
           for name, P in predictors.items()}
    return pd.DataFrame(rows), auc


def assert_well_posed(table, auc):
    # Oracle rank must be ~0 (perfect), Random rank ~0.5; our AUC: Oracle 1.0, Mean ~0.5
    bad = table[table["rank_Oracle"] > 0.05]
    if not bad.empty:
        raise SystemExit(f"WELL-POSEDNESS FAIL: Oracle rank not ~0:\n{bad}")
    rr = table["rank_Random"]
    if not rr.between(0.35, 0.65).all():
        raise SystemExit(f"WELL-POSEDNESS FAIL: Random rank not ~0.5: {rr.tolist()}")
    if auc["Oracle"] < 0.999 or abs(auc["Mean"] - 0.5) > 0.06:
        raise SystemExit(f"WELL-POSEDNESS FAIL: AUC anchors off: {auc}")
    print(f"[gate] Oracle rank~0 OK; Random rank~0.5 OK; AUC Oracle={auc['Oracle']} Mean={auc['Mean']}")


def load_artifact(args):
    if args.pred and args.true and args.meta:
        pred, true = np.load(args.pred), np.load(args.true)
        meta = pd.read_csv(args.meta)
    else:
        res = Path(args.res_dir)
        pred = np.load(res / f"logfc_pred_{args.artifact}.npy")
        true = np.load(res / f"logfc_true_{args.artifact}.npy")
        meta = pd.read_csv(res / f"logfc_meta_{args.artifact}.csv")
    return np.asarray(pred, float), np.asarray(true, float), meta["cell_line"].astype(str).values


def run_real(args):
    score_fn = _import_metric(args.code_dir)
    pred, true, cl = load_artifact(args)
    print(f"[load] pred {pred.shape}, {len(np.unique(cl))} cell lines")
    table, auc = build_table(score_fn, pred, true, cl)
    assert_well_posed(table, auc)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "perturbench_rank_table.csv", index=False)
    pd.DataFrame([auc]).to_csv(out / "control_auc_sidebyside.csv", index=False)
    print("\n=== PerturBench rank (0=best,0.5=rand,1=worst) by framing x distance ===")
    print(table.to_string(index=False))
    print(f"\nour off-diagonal control AUC (within-cell-line, top50 pearson): {auc}")
    print(f"\n[done] -> {out}/perturbench_rank_table.csv + control_auc_sidebyside.csv")
    print("Interpret: a Model rank that is far from 0 (~0.5) AND a control AUC ~0.5 both say "
          "'cannot discriminate'; the within-cellline vs global rank gap is the evidence that "
          "same-cell-line conditioning differs from PerturBench's global rank.")


def run_selftest(args):
    score_fn = _import_metric(args.code_dir)
    rng = np.random.default_rng(13)
    n_cl, n_drug, n_gene = 3, 6, 250
    dsig = rng.normal(0, 1.0, (n_drug, n_gene)); coff = rng.normal(0, 0.5, (n_cl, n_gene))
    true, signal, cl = [], [], []
    for c in range(n_cl):
        for d in range(n_drug):
            true.append(dsig[d] + coff[c] + rng.normal(0, 0.1, n_gene))
            signal.append(dsig[d] + coff[c] + rng.normal(0, 0.4, n_gene))
            cl.append(f"CL{c}")
    true, signal, cl = np.array(true), np.array(signal), np.array(cl)
    table, auc = build_table(score_fn, signal, true, cl)  # 'Model' = planted signal
    print("\n=== SELFTEST rank table (Model = planted signal) ===")
    print(table.to_string(index=False))
    print(f"side-by-side AUC: {auc}")
    assert_well_posed(table, auc)
    sig_ok = bool((table["rank_Model"] < 0.2).all())   # good predictor -> low rank
    print(f"\n[selftest] planted-signal rank < 0.2 in all framings: {sig_ok}")
    if not sig_ok:
        raise SystemExit("SELFTEST FAILED: rank metric not rewarding a good predictor.")
    print("SELFTEST PASSED: PerturBench rank (Eq.2) verified — Oracle~0, Random~0.5, "
          "signal low; AUC anchors Oracle 1.0 / Mean 0.5.")


def main():
    p = argparse.ArgumentParser(description="T3 PerturBench rank vs our control")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--artifact", default="t7_sub_loss_only")
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
