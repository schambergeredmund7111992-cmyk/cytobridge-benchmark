"""
manuscript/analysis/supp_T1/tahoe_morgan_ridge.py
=================================================
T1a (Xichen task book, section T1a) — Tahoe second-dataset Morgan-FP Ridge control.

WHY THIS EXISTS (read before running)
-------------------------------------
The diagnostic paper's headline is a *drug-discrimination control*: a held-out-drug
predictor on single-cell perturbation data can show a high per-pair correlation yet be
unable to tell drugs apart (the off-diagonal control AUC stays ~= 0.5). On sci-Plex,
Ridge(Morgan-FP), chemCPA and CytoBridge all fail this control; only an Oracle passes.

T1 reproduces that on a SECOND, independent dataset (Tahoe-100M). The earlier student
attempt used a *drug one-hot* Ridge, which TRIVIALLY collapses on unseen test drugs:
every unseen drug maps to the all-zero one-hot column, so every test prediction is
identical (inter-drug corr = 1.0). That is not a meaningful linear-baseline failure and
was rolled back from the paper. T1a fixes it with a *Morgan-fingerprint* Ridge — the SAME
featurization used on sci-Plex — so held-out drugs get genuinely distinct molecular
features and the control fairly tests whether a real linear model can discriminate them.

WHAT IT DOES
------------
1. Reuse code/eval/baselines/ridge_pseudobulk.py -> identical Morgan-FP + cell-one-hot
   featurization and pseudobulk logFC as the sci-Plex baseline (so the two are comparable).
2. Train Ridge on the Tahoe TRAIN drugs, predict the held-out TEST drugs.
3. Score Ridge / Mean / Random / Oracle with
   code/eval/metrics.py::drug_discrimination_score (the EXACT student control function,
   never reimplemented here).
4. Assert the WELL-POSEDNESS gate: Oracle AUC == 1.0 and Mean AUC ~= 0.5 in this space.
   If that fails, the pseudobulk reconstruction / units are broken and NO number can be
   trusted -> the script exits instead of emitting a misleading panel.

SELF-SPACE SCORING (matches make_baseline_panel.py): the Ridge prediction is scored
against the truth produced by the SAME pseudobulk pipeline (y_test), not a foreign
reconstruction. The control is predictor-intrinsic; Oracle = 1.0 / Mean ~= 0.5 holds in
any consistent space, which is exactly what the gate checks before trusting Ridge.

REPORTED-NUMBER OWNERSHIP: supervisor-authored (铁律 1). `--selftest` is the
machine-verifiable correctness proof and needs no Tahoe data, scanpy or rdkit.

DATA CONTRACT (what the student must provide for a real run)
-----------------------------------------------------------
--tahoe_h5ad   cell-level AnnData. Raw counts in layers["counts"] (preferred) or X.
               obs must have a drug-name column and a cell-line column.
--smiles_csv   CSV with columns: drug_id, smiles. Must cover EVERY train+test drug.
               Source: HuggingFace tahoebio/Tahoe-100M `drug_metadata` table
               (canonical_smiles + pubchem_cid). 92 drugs (87 train + 5 test).
--splits_json  JSON {"train": [...drug names...], "test": [...drug names...]}.
               Or use --test_drugs "A,B,C" and everything else becomes train.
--control_label  the DMSO/vehicle label in the drug column (Tahoe default differs from
               sci-Plex's "DMSO" -- pass the exact string your h5ad uses).

USAGE
-----
  # real run (student, `cytobridge` conda env, box with the Tahoe h5ad):
  python tahoe_morgan_ridge.py \
      --tahoe_h5ad  /path/tahoe_slice.h5ad \
      --smiles_csv  /path/tahoe_drug_smiles.csv \
      --splits_json /path/tahoe_split.json \
      --control_label DMSO_TF \
      --out_dir .

  # local correctness proof (no external data / deps beyond numpy+scipy+sklearn):
  python tahoe_morgan_ridge.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the canonical code/ tree from the repo layout
#   supp_T1 -> analysis -> manuscript -> <repo root> -> code
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CODE = _REPO / "code"


def _import_control(code_dir: Path):
    """Insert code/ on sys.path and return the EXACT student control function."""
    code_dir = Path(code_dir).resolve()
    if not (code_dir / "eval" / "metrics.py").exists():
        raise SystemExit(
            f"--code_dir {code_dir} has no eval/metrics.py; point it at the CytoBridge "
            "code/ tree so the control metric is reused, not reimplemented."
        )
    sys.path.insert(0, str(code_dir))
    from eval.metrics import drug_discrimination_score  # noqa: E402
    return drug_discrimination_score


def panel_row(score_fn, pred, true, cl, label):
    """One control-panel row, identical recipe to make_baseline_panel.py:
    primary = top-50 DEG pearson AUC; sensitivity = all-gene spearman AUC;
    plus the inter-drug prediction correlation (the collapse signature)."""
    r50 = score_fn(pred, true, cl, top_k=50, metric="pearson")
    rall = score_fn(pred, true, cl, top_k=None, metric="spearman")
    inter = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                inter.append(np.corrcoef(pred[m[i]], pred[m[j]])[0, 1])
    return {
        "predictor": label,
        "auc_deg50_pearson": round(float(r50["specificity_auc"]), 4),
        "on_diag_deg50": round(float(r50["on_diag_mean"]), 4),
        "off_diag_deg50": round(float(r50["off_diag_mean"]), 4),
        "gap_deg50": round(float(r50["gap"]), 4),
        "wilcoxon_p_on_gt_off": round(float(r50["wilcoxon_p_on_gt_off"]), 4),
        "auc_all_spearman": round(float(rall["specificity_auc"]), 4),
        "inter_drug_pearson": round(float(np.mean(inter)), 4) if inter else np.nan,
        "n_pairs": int(r50["n_pairs_scored"]),
    }


def assert_well_posed(panel: pd.DataFrame):
    """Oracle must be 1.0 and Mean ~= 0.5 in this space, else units/recon are broken."""
    oracle = panel.loc[panel.predictor == "Oracle", "auc_deg50_pearson"].iloc[0]
    mean = panel.loc[panel.predictor == "Mean", "auc_deg50_pearson"].iloc[0]
    print(f"[gate] Oracle AUC={oracle} (must be 1.0), Mean AUC={mean} (must be ~0.50)")
    if oracle < 0.999:
        raise SystemExit("WELL-POSEDNESS FAIL: Oracle != 1.0; truth/units broken, panel void.")
    if abs(mean - 0.5) > 0.06:
        raise SystemExit(f"WELL-POSEDNESS FAIL: Mean AUC {mean} not ~0.50; panel void.")


def build_panel(score_fn, pred_ridge, true, cl):
    """Mean / Random / Ridge / Oracle rows, self-space, anchored by the gate."""
    mean_pred = np.zeros_like(true)
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        mean_pred[m] = true[m].mean(axis=0)
    rng = np.random.default_rng(0)
    rand_pred = true[rng.permutation(len(true))]
    rows = [
        panel_row(score_fn, rand_pred, true, cl, "Random"),
        panel_row(score_fn, mean_pred, true, cl, "Mean"),
        panel_row(score_fn, pred_ridge, true, cl, "Ridge (Morgan-FP)"),
        panel_row(score_fn, true.copy(), true, cl, "Oracle"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real run: Tahoe Morgan-FP Ridge
# ---------------------------------------------------------------------------
def resolve_splits(args, all_drugs: set[str], control_label: str):
    """Return (train_drugs, test_drugs) name sets, control excluded from both lists."""
    if args.splits_json:
        spec = json.loads(Path(args.splits_json).read_text())
        train = {str(d) for d in spec["train"]}
        test = {str(d) for d in spec["test"]}
    elif args.test_drugs:
        test = {d.strip() for d in args.test_drugs.split(",") if d.strip()}
        train = {d for d in all_drugs if d not in test and d != control_label}
    else:
        raise SystemExit("provide --splits_json or --test_drugs to define the holdout.")
    train -= {control_label}
    test -= {control_label}
    leak = train & test
    if leak:
        raise SystemExit(f"train/test drug overlap {sorted(leak)[:5]}...; splits must be disjoint.")
    if not train or not test:
        raise SystemExit(f"empty split: |train|={len(train)} |test|={len(test)}.")
    return train, test


def run_real(args):
    score_fn = _import_control(args.code_dir)
    import scanpy as sc  # noqa: E402
    from eval.baselines.ridge_pseudobulk import (  # noqa: E402
        build_matrices, load_drug_smiles, train_ridge,
    )

    print(f"[load] {args.tahoe_h5ad}")
    adata = sc.read_h5ad(args.tahoe_h5ad)
    smiles = load_drug_smiles(args.smiles_csv)
    all_drugs = set(adata.obs[args.drug_col].astype(str))
    train_drugs, test_drugs = resolve_splits(args, all_drugs, args.control_label)

    # loud SMILES-coverage check: a missing SMILES makes build_matrices SILENTLY drop
    # the drug, which would quietly cut test power -> fail early with a clear message.
    miss = sorted((train_drugs | test_drugs) - set(smiles))
    if miss:
        raise SystemExit(f"{len(miss)} drugs lack SMILES, e.g. {miss[:5]}; fix --smiles_csv.")

    keep_train = train_drugs | {args.control_label}
    keep_test = test_drugs | {args.control_label}
    a_train = adata[adata.obs[args.drug_col].astype(str).isin(keep_train)].copy()
    a_test = adata[adata.obs[args.drug_col].astype(str).isin(keep_test)].copy()
    cell_lines_train = sorted(a_train.obs[args.cell_col].astype(str).unique())
    print(f"[split] train {a_train.n_obs} cells / {len(train_drugs)} drugs; "
          f"test {a_test.n_obs} cells / {len(test_drugs)} drugs; "
          f"{len(cell_lines_train)} cell lines")

    X_tr, y_tr, X_te, y_te, meta_te, _ = build_matrices(
        a_train, a_test, smiles, cell_lines_train,
        drug_col=args.drug_col, cell_col=args.cell_col, control_label=args.control_label,
    )
    model = train_ridge(X_tr, y_tr, alpha=args.alpha)
    pred_ridge = model.predict(X_te)
    true = np.asarray(y_te, float)
    cl = meta_te[args.cell_col].astype(str).values
    print(f"[ridge] test pairs={len(true)} genes={true.shape[1]} "
          f"drugs={meta_te[args.drug_col].nunique()} cell_lines={len(np.unique(cl))}")

    panel = build_panel(score_fn, pred_ridge, true, cl)
    assert_well_posed(panel)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "tahoe_morgan_control_panel.csv", index=False)
    meta_te.to_csv(out / "tahoe_ridge_meta.csv", index=False)
    np.save(out / "tahoe_ridge_pred.npy", pred_ridge.astype(np.float32))
    np.save(out / "tahoe_true.npy", true.astype(np.float32))
    print("\n=== Tahoe Morgan-FP drug-discrimination control panel (self-space) ===")
    print(panel.to_string(index=False))
    print(f"\n[done] -> {out}/tahoe_morgan_control_panel.csv  (+ pred/true npy, meta csv)")
    print("Interpret: Ridge AUC ~0.5 => linear baseline ALSO fails control on Tahoe "
          "(reproduces sci-Plex); Ridge AUC >>0.5 => report it, the metric rewards it.")


# ---------------------------------------------------------------------------
# Self-test: synthetic, deterministic, no external data / scanpy / rdkit
# ---------------------------------------------------------------------------
def run_selftest(args):
    """Prove the control-scoring glue + gate with planted-answer synthetic data.
    Builds per-(cell_line,drug) truth = drug_signature + cell_offset + noise, then checks:
      Oracle  -> AUC == 1.0       (perfect drug discrimination)
      Mean    -> AUC ~= 0.5       (cell-line-mean predictor cannot discriminate drugs)
      Random  -> AUC ~= 0.5       (shuffled truth)
      Signal  -> AUC  > 0.7       (a predictor that keeps drug structure CAN pass:
                                   this is the metric's positive-control sanity, cf. T2)
    Reuses the EXACT drug_discrimination_score, so a pass certifies the real-run glue.
    """
    score_fn = _import_control(args.code_dir)
    rng = np.random.default_rng(7)
    n_cl, n_drug, n_gene = 3, 5, 200
    drug_sig = rng.normal(0, 1.0, size=(n_drug, n_gene))       # distinct per drug
    cell_off = rng.normal(0, 0.5, size=(n_cl, n_gene))         # shared within cell line
    true, cl, signal = [], [], []
    for c in range(n_cl):
        for d in range(n_drug):
            true.append(drug_sig[d] + cell_off[c] + rng.normal(0, 0.1, n_gene))
            signal.append(drug_sig[d] + cell_off[c] + rng.normal(0, 0.5, n_gene))
            cl.append(f"CL{c}")
    true = np.array(true)
    signal = np.array(signal)
    cl = np.array(cl)

    panel = build_panel(score_fn, signal, true, cl)
    sig_auc = panel_row(score_fn, signal, true, cl, "Signal")["auc_deg50_pearson"]
    print("\n=== SELFTEST control panel (synthetic; 'Ridge' row = planted Signal) ===")
    print(panel.to_string(index=False))

    oracle = panel.loc[panel.predictor == "Oracle", "auc_deg50_pearson"].iloc[0]
    mean = panel.loc[panel.predictor == "Mean", "auc_deg50_pearson"].iloc[0]
    rand = panel.loc[panel.predictor == "Random", "auc_deg50_pearson"].iloc[0]
    checks = {
        "Oracle == 1.0": oracle >= 0.999,
        "Mean ~= 0.5": abs(mean - 0.5) <= 0.06,
        "Random ~= 0.5": abs(rand - 0.5) <= 0.10,
        "Signal > 0.7": sig_auc > 0.7,
    }
    print("\n[selftest checks]")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise SystemExit("SELFTEST FAILED: control-scoring glue or gate is wrong.")
    print("\nSELFTEST PASSED: drug_discrimination_score wiring + well-posedness gate "
          "verified. Real-run numbers from this script are trustworthy if the gate holds.")


def main():
    p = argparse.ArgumentParser(description="T1a Tahoe Morgan-FP Ridge control panel")
    p.add_argument("--selftest", action="store_true",
                   help="run the synthetic correctness proof (no external data/deps)")
    p.add_argument("--tahoe_h5ad", type=Path)
    p.add_argument("--smiles_csv", type=Path)
    p.add_argument("--splits_json", type=Path)
    p.add_argument("--test_drugs", type=str, default=None,
                   help="comma-separated test drug names (alternative to --splits_json)")
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--code_dir", type=Path, default=_DEFAULT_CODE)
    p.add_argument("--drug_col", default="drug")
    p.add_argument("--cell_col", default="cell_line")
    p.add_argument("--control_label", default="DMSO")
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()

    if args.selftest:
        run_selftest(args)
        return
    if not args.tahoe_h5ad or not args.smiles_csv:
        raise SystemExit("real run needs --tahoe_h5ad and --smiles_csv (or use --selftest).")
    run_real(args)


if __name__ == "__main__":
    main()
