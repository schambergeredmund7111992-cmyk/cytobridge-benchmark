"""
manuscript/analysis/supp_T2/replicate_noise_ceiling.py
======================================================
T2 (Xichen task book, section T2) — positive control / replicate noise ceiling.

WHY (construct validity, the biggest reviewer hole): nothing real has PASSED the
off-diagonal drug-discrimination control yet, so a reviewer can ask "is the control
just too harsh for anything to pass?". This answers it: take a GENUINE biological
replicate. Within each (drug, cell_line), split that condition's cells into two random
halves, pseudobulk each into logFC_A / logFC_B (vs the same-cell-line vehicle), then
score A-as-prediction vs B-as-truth with the SAME control. Real replicate signal SHOULD
pass with AUC >> 0.5 — that is the noise ceiling, and it proves the metric rewards a real
drug-specific predictor (so the collapsed models' ~0.5 is a true negative, not a dead metric).

REUSE, DO NOT REWRITE: scores with code/eval/metrics.py::drug_discrimination_score.
The pseudobulk/logFC recipe matches the rest of the project:
    logFC = log1p(mean treated counts) - log1p(mean same-cell-line vehicle counts).

REPORTED-NUMBER OWNERSHIP: supervisor-authored. The real run needs the cell-level
sci-Plex h5ad (document contract below); `--selftest` proves the split+pseudobulk+score
glue locally with synthetic cells (numpy/scipy only).

DATA CONTRACT (real run): --h5ad a cell-level AnnData with raw counts in
layers["counts"] (preferred) or X, obs has a drug column and a cell-line column, and the
vehicle/control condition is present per cell line (pass its label via --control_label).

USAGE
-----
  python replicate_noise_ceiling.py --selftest
  python replicate_noise_ceiling.py --h5ad sciplex.h5ad \
      --drug_col drug --cell_col cell_line --control_label DMSO --out_dir .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CODE = _REPO / "code"


def _import_metric(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(f"--code_dir {code_dir} has no eval/metrics.py; point it at code/.")
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def replicate_halves_from_cells(X, drugs, cells, control_label, seed=0, min_cells=10):
    """Split each (drug, cell_line) condition's cells into two halves and build the
    replicate logFC pair (A, B) vs the shared same-cell-line vehicle pseudobulk.
    Pure numpy so the selftest needs no scanpy/anndata. Returns predA, trueB, cl, drugs."""
    X = np.asarray(X, float)
    drugs = np.asarray(drugs).astype(str)
    cells = np.asarray(cells).astype(str)
    rng = np.random.default_rng(seed)
    control = {}
    for c in np.unique(cells):
        m = (cells == c) & (drugs == control_label)
        if m.any():
            control[c] = np.log1p(X[m].mean(axis=0))
    predA, trueB, cl_out, drug_out = [], [], [], []
    for c in np.unique(cells):
        if c not in control:
            continue
        for d in np.unique(drugs[cells == c]):
            if d == control_label:
                continue
            idx = np.flatnonzero((cells == c) & (drugs == d))
            if idx.size < min_cells:
                continue
            idx = rng.permutation(idx)
            ha, hb = idx[: idx.size // 2], idx[idx.size // 2:]
            la = np.log1p(X[ha].mean(axis=0)) - control[c]
            lb = np.log1p(X[hb].mean(axis=0)) - control[c]
            predA.append(la); trueB.append(lb); cl_out.append(c); drug_out.append(d)
    if not predA:
        raise SystemExit("no condition had >= min_cells treated cells to split; "
                         "lower --min_cells or check drug/cell columns + control_label.")
    return (np.array(predA), np.array(trueB), np.array(cl_out), np.array(drug_out))


def panel_row(score_fn, pred, true, cl, label):
    r = score_fn(pred, true, cl, top_k=50, metric="pearson")
    return {"predictor": label,
            "auc_deg50_pearson": round(float(r["specificity_auc"]), 4),
            "gap_deg50": round(float(r["gap"]), 4),
            "wilcoxon_p_on_gt_off": round(float(r["wilcoxon_p_on_gt_off"]), 4),
            "n_pairs": int(r["n_pairs_scored"])}


def build_panel(score_fn, predA, trueB, cl):
    mean_pred = np.zeros_like(trueB)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        mean_pred[m] = trueB[m].mean(axis=0)
    return pd.DataFrame([
        panel_row(score_fn, mean_pred, trueB, cl, "Mean (neg ctrl)"),
        panel_row(score_fn, predA, trueB, cl, "Replicate A-vs-B (noise ceiling)"),
        panel_row(score_fn, trueB.copy(), trueB, cl, "Oracle (pos ctrl)"),
    ])


def assert_well_posed(panel):
    o = panel.loc[panel.predictor.str.startswith("Oracle"), "auc_deg50_pearson"].iloc[0]
    mn = panel.loc[panel.predictor.str.startswith("Mean"), "auc_deg50_pearson"].iloc[0]
    print(f"[gate] Oracle AUC={o} (must 1.0), Mean AUC={mn} (must ~0.5)")
    if o < 0.999:
        raise SystemExit("WELL-POSEDNESS FAIL: Oracle != 1.0.")
    if abs(mn - 0.5) > 0.06:
        raise SystemExit(f"WELL-POSEDNESS FAIL: Mean AUC {mn} not ~0.5.")


def run_real(args):
    score_fn = _import_metric(args.code_dir)
    import scanpy as sc  # noqa: E402
    adata = sc.read_h5ad(args.h5ad)
    X = adata.layers["counts"] if ("counts" in adata.layers) else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    drugs = adata.obs[args.drug_col].values
    cells = adata.obs[args.cell_col].values
    predA, trueB, cl, drug = replicate_halves_from_cells(
        X, drugs, cells, args.control_label, seed=args.seed, min_cells=args.min_cells)
    print(f"[replicate] {len(cl)} conditions, {len(np.unique(drug))} drugs, "
          f"{len(np.unique(cl))} cell lines")
    panel = build_panel(score_fn, predA, trueB, cl)
    assert_well_posed(panel)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "replicate_noise_ceiling_panel.csv", index=False)
    print("\n=== replicate noise-ceiling control panel ===")
    print(panel.to_string(index=False))
    print(f"\n[done] -> {out}/replicate_noise_ceiling_panel.csv")
    print("Interpret: Replicate AUC >> 0.5 => the metric DOES reward a real "
          "drug-specific predictor, so the models' ~0.5 is a true collapse, not a dead metric.")


def _synth_cells(rng, n_cl, n_drug, n_gene, cells_per, signal=True):
    """Synthetic counts: per-(cell,drug) mean signal + Poisson-ish cell noise, plus a
    vehicle condition per cell line. signal=False makes all drugs share the control mean."""
    base = rng.uniform(1.0, 4.0, n_gene)                      # control log-mean
    drug_eff = rng.normal(0, 1.2, (n_drug, n_gene)) if signal else np.zeros((n_drug, n_gene))
    cell_eff = rng.normal(0, 0.3, (n_cl, n_gene))
    X, drugs, cells = [], [], []
    for c in range(n_cl):
        ctrl_mean = np.exp(base + cell_eff[c])
        for _ in range(cells_per):
            X.append(rng.poisson(ctrl_mean)); drugs.append("DMSO"); cells.append(f"CL{c}")
        for d in range(n_drug):
            cond_mean = np.exp(base + cell_eff[c] + drug_eff[d])
            for _ in range(cells_per):
                X.append(rng.poisson(cond_mean)); drugs.append(f"D{d}"); cells.append(f"CL{c}")
    return np.array(X, float), np.array(drugs), np.array(cells)


def run_selftest(args):
    score_fn = _import_metric(args.code_dir)
    rng = np.random.default_rng(5)
    # signal case: real per-drug effect -> replicate halves should discriminate
    X, drugs, cells = _synth_cells(rng, 3, 6, 200, cells_per=60, signal=True)
    predA, trueB, cl, _ = replicate_halves_from_cells(X, drugs, cells, "DMSO", seed=1, min_cells=10)
    panel = build_panel(score_fn, predA, trueB, cl)
    print("\n=== SELFTEST panel (signal case) ===")
    print(panel.to_string(index=False))
    assert_well_posed(panel)
    rep_auc = panel.loc[panel.predictor.str.startswith("Replicate"), "auc_deg50_pearson"].iloc[0]

    # no-signal case: all drugs == control distribution -> replicate cannot discriminate
    Xn, dn, cn = _synth_cells(np.random.default_rng(6), 3, 6, 200, cells_per=60, signal=False)
    pa, tb, cl2, _ = replicate_halves_from_cells(Xn, dn, cn, "DMSO", seed=2, min_cells=10)
    null_auc = panel_row(score_fn, pa, tb, cl2, "rep")["auc_deg50_pearson"]

    checks = {
        "Replicate(signal) AUC > 0.7": rep_auc > 0.7,
        "Replicate(no-signal) AUC ~ 0.5": abs(null_auc - 0.5) <= 0.12,
    }
    print(f"\n[selftest] replicate signal AUC={rep_auc}, no-signal AUC={null_auc}")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        raise SystemExit("SELFTEST FAILED: noise-ceiling glue wrong.")
    print("SELFTEST PASSED: split+pseudobulk+control scoring verified; the metric "
          "rewards a real replicate and stays ~0.5 with no signal.")


def main():
    p = argparse.ArgumentParser(description="T2 replicate noise-ceiling positive control")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--h5ad", type=Path)
    p.add_argument("--drug_col", default="drug")
    p.add_argument("--cell_col", default="cell_line")
    p.add_argument("--control_label", default="DMSO")
    p.add_argument("--min_cells", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--code_dir", type=Path, default=_DEFAULT_CODE)
    args = p.parse_args()
    if args.selftest:
        run_selftest(args)
    elif args.h5ad:
        run_real(args)
    else:
        raise SystemExit("real run needs --h5ad (or use --selftest).")


if __name__ == "__main__":
    main()
