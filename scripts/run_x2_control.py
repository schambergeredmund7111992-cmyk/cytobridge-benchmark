"""
X2: Tahoe external dataset control
----------------------------------
After build_external_split.py has created the drug-disjoint split, this script:
1. Loads the test split counts + manifest
2. Computes pseudobulk logFC for test pairs
3. Runs Mean, Ridge (Morgan FP + cell one-hot), and Oracle controls
4. Saves prediction vectors + meta for downstream tabulation

Usage:
    source /path/to/venvs/cytobridge-scgpt-py310/bin/activate
    cd /path/to/cytobridge-benchmark
    python scripts/run_x2_control.py \
        --split_dir data/processed/tahoe/splits_external \
        --out_dir results/tahoe_control
"""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from eval.metrics import drug_discrimination_score


def compute_pseudobulk_logfc(treated_counts, control_counts, manifest, drug_col="drug_id"):
    """Group cells by (drug, cell_line), compute pseudobulk logFC."""
    groups = defaultdict(lambda: {"treated": [], "control": []})
    for i in range(len(manifest)):
        drug = manifest.iloc[i][drug_col]
        cell = manifest.iloc[i]["cell_line"]
        groups[(drug, cell)]["treated"].append(treated_counts[i])
        groups[(drug, cell)]["control"].append(control_counts[i])

    pairs = []
    logfc = []
    for (drug, cell) in sorted(groups.keys()):
        t_pb = np.mean(groups[(drug, cell)]["treated"], axis=0)
        c_pb = np.mean(groups[(drug, cell)]["control"], axis=0)
        logfc.append(np.log1p(t_pb) - np.log1p(c_pb))
        pairs.append((drug, cell))

    return np.array(logfc), pairs


def mean_predictor(true_logfc, cell_lines):
    """Cell-line mean predictor — identical for every drug."""
    pred = np.zeros_like(true_logfc)
    for cl in np.unique(cell_lines):
        m = np.flatnonzero(cell_lines == cl)
        pred[m] = true_logfc[m].mean(axis=0)
    return pred


def ridge_predictor(train_treated_counts, train_control_counts, train_manifest, test_manifest,
                    smiles_csv=None, n_genes=3000):
    """
    Ridge regression: Morgan fingerprint (1024-bit) + cell one-hot -> gene expression.
    If SMILES not available, use drug_id as categorical feature.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Build Morgan fingerprints for each drug
    drug_to_fp = {}
    if smiles_csv and os.path.exists(smiles_csv):
        sm_df = pd.read_csv(smiles_csv)
        for _, row in sm_df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                drug_to_fp[row["drug_id"]] = np.array(fp, dtype=float)
    else:
        print("[ridge] No SMILES CSV; using drug_id one-hot as features")
        all_drugs = sorted(set(train_manifest["drug_id"].unique()) |
                          set(test_manifest["drug_id"].unique()))
        for i, d in enumerate(all_drugs):
            vec = np.zeros(len(all_drugs))
            vec[i] = 1.0
            drug_to_fp[d] = vec

    cell_lines_all = sorted(set(train_manifest["cell_line"].unique()) |
                           set(test_manifest["cell_line"].unique()))
    cell_encoder = OneHotEncoder(sparse_output=False)
    cell_encoder.fit(np.array(cell_lines_all).reshape(-1, 1))

    def build_X(manifest):
        X_rows = []
        for _, row in manifest.iterrows():
            drug_vec = drug_to_fp.get(row["drug_id"], np.zeros(1024))
            cell_vec = cell_encoder.transform([[row["cell_line"]]])[0]
            X_rows.append(np.concatenate([drug_vec, cell_vec]))
        return np.array(X_rows)

    # Aggregate train data to pseudobulk per (drug, cell_line)
    train_groups = defaultdict(lambda: {"treated": [], "control": []})
    for i in range(len(train_manifest)):
        d = train_manifest.iloc[i]["drug_id"]
        cl = train_manifest.iloc[i]["cell_line"]
        train_groups[(d, cl)]["treated"].append(train_treated_counts[i])
        train_groups[(d, cl)]["control"].append(train_control_counts[i])

    X_train_rows = []
    y_train_rows = []
    for (d, cl), data in sorted(train_groups.items()):
        t_pb = np.mean(data["treated"], axis=0)
        c_pb = np.mean(data["control"], axis=0)
        y_train_rows.append(np.log1p(t_pb) - np.log1p(c_pb))
        drug_vec = drug_to_fp.get(d, np.zeros(1024))
        cell_vec = cell_encoder.transform([[cl]])[0]
        X_train_rows.append(np.concatenate([drug_vec, cell_vec]))

    X_train = np.array(X_train_rows)
    y_train = np.array(y_train_rows)

    print(f"[ridge] Training: X={X_train.shape}, y={y_train.shape}")
    print(f"[ridge] Drugs with FPs: {len(drug_to_fp)}")

    # Fit Ridge per gene (or use multi-output Ridge)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict on test pairs
    test_groups = defaultdict(int)
    for i in range(len(test_manifest)):
        d = test_manifest.iloc[i]["drug_id"]
        cl = test_manifest.iloc[i]["cell_line"]
        test_groups[(d, cl)] += 1

    X_test_rows = []
    test_pairs = []
    for (d, cl) in sorted(test_groups.keys()):
        drug_vec = drug_to_fp.get(d, np.zeros(1024))
        cell_vec = cell_encoder.transform([[cl]])[0]
        X_test_rows.append(np.concatenate([drug_vec, cell_vec]))
        test_pairs.append((d, cl))

    X_test = np.array(X_test_rows)
    ridge_pred = model.predict(X_test)
    return ridge_pred, test_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=Path, required=True,
                        help="Directory with tahoe_external_*.parquet and count files")
    parser.add_argument("--out_dir", type=Path, default=Path("results/tahoe_control"),
                        help="Output directory")
    parser.add_argument("--smiles_csv", type=Path, default=None,
                        help="Path to SMILES CSV (optional)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    test_manifest = pd.read_parquet(args.split_dir / "tahoe_external_test.parquet")
    test_treated = np.load(args.split_dir / "tahoe_external_test_treated_counts.npy")
    test_control = np.load(args.split_dir / "tahoe_external_test_control_counts.npy")

    print(f"Test: {len(test_manifest)} cells, "
          f"{test_manifest['drug_id'].nunique()} drugs, "
          f"{test_manifest['cell_line'].nunique()} cell lines")

    # Compute pseudobulk logFC
    true_logfc, pairs = compute_pseudobulk_logfc(test_treated, test_control, test_manifest)
    n_pairs = len(pairs)
    n_genes = true_logfc.shape[1]
    cell_lines = np.array([p[1] for p in pairs])

    print(f"Test pairs: {n_pairs}, genes: {n_genes}")

    # Save true vectors
    np.save(out_dir / "logfc_true_tahoe.npy", true_logfc)
    pd.DataFrame(pairs, columns=["drug", "cell_line"]).to_csv(
        out_dir / "logfc_meta_tahoe.csv", index=False)

    results = []

    # ---- Oracle (positive control) ----
    oracle_res = drug_discrimination_score(true_logfc, true_logfc, cell_lines,
                                           top_k=50, metric="pearson")
    results.append({"predictor": "Oracle", "auc": oracle_res["specificity_auc"],
                    "gap": oracle_res["gap"], "on_diag": oracle_res["on_diag_mean"],
                    "rho50": 1.0})
    print(f"Oracle: AUC={oracle_res['specificity_auc']:.3f}")

    # ---- Mean (negative control) ----
    mean_pred = mean_predictor(true_logfc, cell_lines)
    mean_res = drug_discrimination_score(mean_pred, true_logfc, cell_lines,
                                         top_k=50, metric="pearson")
    rho50_mean = np.mean([stats.spearmanr(true_logfc[i][topk], mean_pred[i][topk]).statistic
                          for i in range(n_pairs)
                          if (topk := np.argsort(-np.abs(true_logfc[i]))[:50]) is not None])
    results.append({"predictor": "Mean", "auc": mean_res["specificity_auc"],
                    "gap": mean_res["gap"], "on_diag": mean_res["on_diag_mean"],
                    "rho50": rho50_mean})
    np.save(out_dir / "pred_mean_tahoe.npy", mean_pred)
    print(f"Mean: AUC={mean_res['specificity_auc']:.3f}, rho50={rho50_mean:.4f}")

    # ---- Ridge ----
    try:
        train_manifest = pd.read_parquet(args.split_dir / "tahoe_external_train.parquet")
        train_treated = np.load(args.split_dir / "tahoe_external_train_treated_counts.npy")
        train_control = np.load(args.split_dir / "tahoe_external_train_control_counts.npy")

        ridge_pred, ridge_pairs = ridge_predictor(
            train_treated, train_control, train_manifest, test_manifest,
            smiles_csv=args.smiles_csv, n_genes=n_genes)

        # Align ridge predictions to test order
        ridge_aligned = np.zeros_like(true_logfc)
        ridge_order = {p: i for i, p in enumerate(ridge_pairs)}
        for i, p in enumerate(pairs):
            if p in ridge_order:
                ridge_aligned[i] = ridge_pred[ridge_order[p]]
            else:
                ridge_aligned[i] = np.zeros(n_genes)

        ridge_res = drug_discrimination_score(ridge_aligned, true_logfc, cell_lines,
                                              top_k=50, metric="pearson")
        topk_idx = [np.argsort(-np.abs(true_logfc[i]))[:50] for i in range(n_pairs)]
        rho50_ridge = np.mean([stats.spearmanr(true_logfc[i][topk_idx[i]],
                                                ridge_aligned[i][topk_idx[i]]).statistic
                               for i in range(n_pairs)])
        results.append({"predictor": "Ridge", "auc": ridge_res["specificity_auc"],
                        "gap": ridge_res["gap"], "on_diag": ridge_res["on_diag_mean"],
                        "rho50": rho50_ridge})
        np.save(out_dir / "pred_ridge_tahoe.npy", ridge_aligned)
        print(f"Ridge: AUC={ridge_res['specificity_auc']:.3f}, rho50={rho50_ridge:.4f}")
    except Exception as e:
        print(f"[ridge] FAILED: {e}")
        results.append({"predictor": "Ridge", "auc": float("nan"),
                        "gap": float("nan"), "on_diag": float("nan"),
                        "rho50": float("nan")})

    # Save results table
    panel = pd.DataFrame(results)
    panel.to_csv(out_dir / "tahoe_control_panel.csv", index=False)
    print("\n=== Tahoe Control Panel ===")
    print(panel.to_string(index=False))
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
