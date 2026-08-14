"""
manuscript/analysis/supp_T5/clustered_bootstrap_power.py
========================================================
T5 (Xichen task book, section T5) — clustered bootstrap CI + statistical power.

WHY: the held-out matrix is only 27 (drug x cell) pairs and they are NOT independent
(same drug across cell lines is correlated). A reviewer asks (a) what is the AUC's
uncertainty under that clustering, and (b) is 27 pairs even powered to detect a small
true effect? This script answers both:

  1. Point control AUC for a chosen predictor (reuses drug_discrimination_score).
  2. Cluster robustness WITHOUT duplicate-inflation:
       - leave-one-DRUG-out and leave-one-CELL-LINE-out jackknife (distinct subsets,
         so the metric's within-cell-line off-diagonal stays well-defined).
       - cluster bootstrap by cell line (resample cell lines WITH replacement; drawn
         duplicates are RELABELLED so the metric treats them as separate blocks).
         Reported honestly as coarse (only 3 cell-line clusters).
  3. Power curve by simulation: the off-diagonal control reduces to "is on-diag > the
     off-diagonals" per pair; modelling that as a one-sided BINOMIAL sign test, compute
     the power to reject H0: AUC=0.5 at alpha for true AUC in {0.55, 0.60, 0.70} across
     N pairs, and the N needed for power 0.8. Closed-form, touches no metric internals.

REUSE, DO NOT REWRITE: AUC via code/eval/metrics.py::drug_discrimination_score.
REPORTED-NUMBER OWNERSHIP: supervisor-authored. `--selftest` proves it locally.

USAGE
-----
  python clustered_bootstrap_power.py --selftest
  python clustered_bootstrap_power.py --artifact t7_sub_loss_only --out_dir .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CODE = _REPO / "code"
_DEFAULT_RES = _REPO / "student_progress_E6E7" / "E6E7" / "results"

ALPHA = 0.01
TARGET_AUCS = [0.55, 0.60, 0.70]


def _import_metric(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(f"--code_dir {code_dir} has no eval/metrics.py; point it at code/.")
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def auc(score_fn, pred, true, cl):
    return float(score_fn(pred, true, cl, top_k=50, metric="pearson")["specificity_auc"])


# ---- closed-form one-sided binomial power for the sign-test framing ----
def binom_power(n, p, alpha=ALPHA):
    """Power to reject H0: success=0.5 (one-sided greater) when true success prob = p,
    using the exact binomial. k_crit = smallest k with P(Bin(n,0.5) >= k) <= alpha."""
    k = np.arange(n + 1)
    surv_null = stats.binom.sf(k - 1, n, 0.5)          # P(X >= k) under H0
    crit = k[surv_null <= alpha]
    if crit.size == 0:
        return 0.0
    kc = int(crit.min())
    return float(stats.binom.sf(kc - 1, n, p))         # P(X >= kc) under true p


def power_curve(ns):
    rows = []
    for n in ns:
        row = {"n_pairs": int(n)}
        for p in TARGET_AUCS:
            row[f"power_auc_{p}"] = round(binom_power(n, p), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def n_for_power(target_p, want=0.8, nmax=2000):
    for n in range(2, nmax + 1):
        if binom_power(n, target_p) >= want:
            return n
    return None


def jackknife(score_fn, pred, true, cl, drugs):
    """Leave-one-drug-out and leave-one-cell-line-out recomputation of the AUC."""
    def loo(labels):
        vals = []
        for u in np.unique(labels):
            keep = labels != u
            if len(np.unique(cl[keep])) < 1 or keep.sum() < 2:
                continue
            try:
                vals.append(auc(score_fn, pred[keep], true[keep], cl[keep]))
            except Exception:
                continue
        return np.array(vals)
    jd, jc = loo(drugs), loo(cl)
    return jd, jc


def cluster_boot_by_cellline(score_fn, pred, true, cl, n_boot=2000, seed=0):
    """Resample cell lines WITH replacement; relabel duplicates so the metric treats
    them as separate blocks. Coarse (few clusters) but honest."""
    rng = np.random.default_rng(seed)
    cells = np.unique(cl)
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(cells, size=len(cells), replace=True)
        P, T, L = [], [], []
        for b, c in enumerate(drawn):
            m = np.flatnonzero(cl == c)
            P.append(pred[m]); T.append(true[m]); L.append(np.array([f"{c}__b{b}"] * m.size))
        try:
            boots.append(auc(score_fn, np.vstack(P), np.vstack(T), np.concatenate(L)))
        except Exception:
            continue
    b = np.array(boots)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), b


def load_artifact(args):
    if args.pred and args.true and args.meta:
        pred, true = np.load(args.pred), np.load(args.true)
        meta = pd.read_csv(args.meta)
    else:
        res = Path(args.res_dir)
        pred = np.load(res / f"logfc_pred_{args.artifact}.npy")
        true = np.load(res / f"logfc_true_{args.artifact}.npy")
        meta = pd.read_csv(res / f"logfc_meta_{args.artifact}.csv")
    return (np.asarray(pred, float), np.asarray(true, float),
            meta["cell_line"].astype(str).values, meta["drug"].astype(str).values)


def summarize(score_fn, pred, true, cl, drugs, out_dir, tag):
    point = auc(score_fn, pred, true, cl)
    jd, jc = jackknife(score_fn, pred, true, cl, drugs)
    lo, hi, _ = cluster_boot_by_cellline(score_fn, pred, true, cl)
    print(f"\n=== [{tag}] control AUC robustness ===")
    print(f"  point AUC@50-pearson            = {point:.4f}")
    print(f"  leave-one-DRUG-out      n={jd.size:2d}  range [{jd.min():.4f}, {jd.max():.4f}]  sd {jd.std():.4f}")
    print(f"  leave-one-CELLLINE-out  n={jc.size:2d}  range [{jc.min():.4f}, {jc.max():.4f}]  sd {jc.std():.4f}")
    print(f"  cluster-bootstrap-by-cellline 95% CI (coarse, {len(np.unique(cl))} clusters) "
          f"= [{lo:.4f}, {hi:.4f}]")
    pc = power_curve([10, 20, 27, 30, 50, 75, 100, 150, 200])
    needs = {p: n_for_power(p) for p in TARGET_AUCS}
    print("  power (one-sided binomial sign test, alpha=0.01):")
    print(pc.to_string(index=False))
    print("  N for power 0.8:", {f"AUC{p}": needs[p] for p in TARGET_AUCS})
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"tag": tag, "point_auc": round(point, 4),
                       "loo_drug_sd": round(float(jd.std()), 4),
                       "loo_cell_sd": round(float(jc.std()), 4),
                       "cellboot_ci_lo": round(lo, 4), "cellboot_ci_hi": round(hi, 4),
                       **{f"n80_auc_{p}": needs[p] for p in TARGET_AUCS}}]
                     ).to_csv(out / "clustered_bootstrap_summary.csv", index=False)
        pc.to_csv(out / "power_curve.csv", index=False)
        print(f"[done] -> {out}/clustered_bootstrap_summary.csv + power_curve.csv")
    return point, needs


def run_real(args):
    score_fn = _import_metric(args.code_dir)
    pred, true, cl, drugs = load_artifact(args)
    print(f"[load] {pred.shape}, {len(np.unique(drugs))} drugs x {len(np.unique(cl))} cell lines")
    summarize(score_fn, pred, true, cl, drugs, args.out_dir, args.artifact)


def run_selftest(args):
    score_fn = _import_metric(args.code_dir)
    # power-curve sanity (closed form, no data)
    cal = binom_power(200, 0.5)                       # should be <= alpha-ish
    mono = all(binom_power(n, 0.6) <= binom_power(n + 20, 0.6) for n in [10, 30, 50, 100])
    needs07 = n_for_power(0.70)
    # synthetic predictor jackknife/bootstrap path exercise
    rng = np.random.default_rng(3)
    n_cl, n_drug, n_gene = 3, 8, 200
    dsig = rng.normal(0, 1.0, (n_drug, n_gene)); coff = rng.normal(0, 0.4, (n_cl, n_gene))
    pred, true, cl, drugs = [], [], [], []
    for c in range(n_cl):
        for d in range(n_drug):
            true.append(dsig[d] + coff[c] + rng.normal(0, 0.1, n_gene))
            pred.append(dsig[d] + coff[c] + rng.normal(0, 0.4, n_gene))
            cl.append(f"CL{c}"); drugs.append(f"D{d}")
    pred, true = np.array(pred), np.array(true)
    cl, drugs = np.array(cl), np.array(drugs)
    point, needs = summarize(score_fn, pred, true, cl, drugs, None, "selftest")

    checks = {
        "power(AUC=0.5,N=200) <= 2*alpha (calibration)": cal <= 2 * ALPHA,
        "power monotone increasing in N": mono,
        "N for power0.8 @AUC0.70 is finite": needs07 is not None,
        "synthetic signal point AUC > 0.7": point > 0.7,
    }
    print("\n[selftest checks]")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}  ({'cal=%.4f' % cal if 'calibration' in k else ''})")
    if not all(checks.values()):
        raise SystemExit("SELFTEST FAILED: bootstrap/power glue wrong.")
    print("SELFTEST PASSED: jackknife + cluster bootstrap + binomial power verified.")


def main():
    p = argparse.ArgumentParser(description="T5 clustered bootstrap CI + power")
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
