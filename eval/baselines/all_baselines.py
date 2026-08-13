#!/usr/bin/env python
"""All baselines comparison table.
Produces: {Mean, Linear, Linear-adj, Ridge_clean, kNN-MolFormer} x {R2_all, R2_DEG, Spearman}

Uses per-split npy files + parquet manifests (no full h5ad needed at runtime).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits)
    for i, bit in enumerate(fp):
        if bit:
            arr[i] = 1.0
    return arr


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Pseudobulk construction
# ---------------------------------------------------------------------------
def build_pseudobulk(manifest_path, treated_path, control_path):
    """Return dict {(drug, cell_line): {log_fc: [G], n_treated: int}}."""
    man = pd.read_parquet(manifest_path).reset_index(drop=True)
    treated = np.load(treated_path)
    control = np.load(control_path)
    pairs = {}
    for (drug, cl), idx in man.groupby(["drug_id", "cell_line"]).groups.items():
        take = np.asarray(list(idx), dtype=int)
        treated_pb = treated[take].mean(axis=0)
        ctrl_pb = control[take].mean(axis=0)
        log_fc = np.log1p(np.maximum(treated_pb, 0)) - np.log1p(np.maximum(ctrl_pb, 0))
        pairs[(str(drug), str(cl))] = {"log_fc": log_fc.astype(np.float64),
                                        "n_treated": len(take)}
    return pairs


def merge_pairs(*dicts):
    """Merge multiple pair dicts; later dicts overwrite earlier on key conflict."""
    out = {}
    for d in dicts:
        out.update(d)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(pairs_dict, preds_dict, top_k=50):
    """Per-gene R2(all), per-gene R2(top-K DEG), per-pair Spearman(top-K DEG).

    R2 is computed per-gene across all test pairs (field standard: chemCPA/PerturbNet).
    Spearman is per-pair to match ridge baseline comparison.
    """
    keys = list(pairs_dict.keys())
    n_genes = len(pairs_dict[keys[0]]["log_fc"])
    n_pairs = len(keys)

    # Collect [n_pairs, n_genes] matrices
    true_mat = np.stack([pairs_dict[k]["log_fc"] for k in keys])
    pred_mat = np.stack([preds_dict.get(k, np.zeros(n_genes)) for k in keys])

    # Per-gene R2 (all genes)
    ss_res = ((true_mat - pred_mat) ** 2).sum(axis=0)
    ss_tot = ((true_mat - true_mat.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_gene = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0)
    r2_all = float(r2_per_gene.mean())

    # Per-pair Spearman on top-K DEG (matching ridge eval)
    spear_list = []
    for i, key in enumerate(keys):
        true = true_mat[i]
        pred = pred_mat[i]
        if np.std(true) < 1e-6 or np.std(pred) < 1e-6:
            continue
        top_idx = np.argsort(-np.abs(true))[:top_k]
        sr = spearmanr(pred[top_idx], true[top_idx])[0]
        spear_list.append(0.0 if np.isnan(sr) else float(sr))

    # Per-gene R2 on union of per-pair top-K DEG genes
    deg_union = set()
    for i in range(n_pairs):
        top_idx = np.argsort(-np.abs(true_mat[i]))[:top_k]
        deg_union.update(top_idx.tolist())
    deg_genes = sorted(deg_union)
    r2_deg = float(r2_per_gene[deg_genes].mean()) if deg_genes else float("nan")

    return {
        "R2_all": r2_all,
        "R2_DEG": r2_deg,
        "Spearman_DEG": float(np.mean(spear_list)) if spear_list else float("nan"),
        "n_pairs": n_pairs,
        "n_deg_genes": len(deg_genes),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_mean(fit_pairs, test_pairs, cell_lines, n_genes=3000):
    """Per-cell-line mean logFC over fit drugs."""
    cl_mean = {}
    for cl in cell_lines:
        vals = [info["log_fc"] for (d, c), info in fit_pairs.items() if c == cl]
        cl_mean[cl] = np.mean(vals, axis=0) if vals else np.zeros(n_genes)
    return {(d, cl): cl_mean.get(cl, np.zeros(n_genes)) for (d, cl) in test_pairs}


def _fit_per_gene_ridge(train_X, train_Y, alpha):
    n_genes = train_Y.shape[1]
    scaler = StandardScaler().fit(train_X)
    X_s = scaler.transform(train_X)
    models = []
    for g in range(n_genes):
        y = train_Y[:, g]
        if np.std(y) < 1e-8:
            models.append(("const", float(np.mean(y))))
        else:
            m = Ridge(alpha=alpha, fit_intercept=True, max_iter=5000)
            m.fit(X_s, y)
            models.append(("ridge", m))
    return models, scaler


def _predict_per_gene(models, scaler, X):
    X_s = scaler.transform(X)
    n = X_s.shape[0]
    n_genes = len(models)
    pred = np.zeros((n, n_genes))
    for g, (kind, m) in enumerate(models):
        pred[:, g] = m if kind == "const" else m.predict(X_s)
    return pred


def baseline_linear(fit_pairs, test_pairs, fp_cache, cell_lines, alpha=1.0):
    """Ridge on Morgan FP only (no cell-line feature)."""
    fit_keys = list(fit_pairs)
    train_X = np.stack([fp_cache[d] for d, _ in fit_keys])
    train_Y = np.stack([fit_pairs[k]["log_fc"] for k in fit_keys])
    models, scaler = _fit_per_gene_ridge(train_X, train_Y, alpha)

    test_keys = list(test_pairs)
    test_X = np.stack([fp_cache[d] for d, _ in test_keys])
    pred_all = _predict_per_gene(models, scaler, test_X)
    return {k: pred_all[i] for i, k in enumerate(test_keys)}


def baseline_linear_adjusted(fit_pairs, test_pairs, fp_cache, cell_lines, alpha=1.0):
    """Ridge on Morgan FP + cell-line one-hot (= clean ridge)."""
    fit_keys = list(fit_pairs)
    cell_enc = OneHotEncoder(sparse_output=False).fit(np.array(cell_lines).reshape(-1, 1))

    train_fp = np.stack([fp_cache[d] for d, _ in fit_keys])
    train_cell = cell_enc.transform(np.array([c for _, c in fit_keys]).reshape(-1, 1))
    train_X = np.hstack([train_fp, train_cell])
    train_Y = np.stack([fit_pairs[k]["log_fc"] for k in fit_keys])
    models, scaler = _fit_per_gene_ridge(train_X, train_Y, alpha)

    test_keys = list(test_pairs)
    test_fp = np.stack([fp_cache[d] for d, _ in test_keys])
    test_cell = cell_enc.transform(np.array([c for _, c in test_keys]).reshape(-1, 1))
    test_X = np.hstack([test_fp, test_cell])
    pred_all = _predict_per_gene(models, scaler, test_X)
    return {k: pred_all[i] for i, k in enumerate(test_keys)}


def baseline_knn_molformer(fit_pairs, test_pairs, molformer_emb, molformer_ids,
                           cell_lines, k=5, temperature=5.0):
    """kNN in MolFormer space: cosine-weighted average of k nearest fit drugs' per-cell-line delta.

    For each test drug, finds k nearest fit drugs, then for each cell line averages
    their per-cell-line pseudobulk delta weighted by cosine similarity.
    """
    n_genes = 3000
    emb = molformer_emb.mean(axis=1)  # [N_drugs, 768] mean pool over tokens
    id_to_vec = {}
    for i, d in enumerate(molformer_ids):
        v = emb[i]
        id_to_vec[str(d)] = v / (np.linalg.norm(v) + 1e-12)

    # Per-drug per-cell-line delta from fit pairs
    fit_drugs = sorted({d for d, _ in fit_pairs})
    fit_delta = {}  # drug -> {cell_line -> delta}
    for drug in fit_drugs:
        fit_delta[drug] = {}
        for cl in cell_lines:
            key = (drug, cl)
            fit_delta[drug][cl] = fit_pairs[key]["log_fc"] if key in fit_pairs else np.zeros(n_genes)

    # Normalized fit drug vectors
    fit_vecs = np.stack([id_to_vec.get(d, np.zeros(768)) for d in fit_drugs])

    test_drugs = sorted({d for d, _ in test_pairs})
    preds = {}
    for drug in test_drugs:
        q = id_to_vec.get(drug)
        if q is None:
            for cl in cell_lines:
                preds[(drug, cl)] = np.zeros(n_genes)
            continue
        sim = fit_vecs @ q
        top_k = np.argsort(-sim)[:k]
        w = np.exp(sim[top_k] * temperature)
        w /= w.sum()

        for cl in cell_lines:
            delta = np.zeros(n_genes)
            for i, weight in zip(top_k, w):
                delta += weight * fit_delta[fit_drugs[i]].get(cl, np.zeros(n_genes))
            preds[(drug, cl)] = delta
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="data/processed/sciplex/splits")
    ap.add_argument("--splits_json", default="data/processed/sciplex/splits/internal_splits.json")
    ap.add_argument("--smiles_csv", default="data/processed/sciplex/drugs_canonical.csv")
    ap.add_argument("--molformer_npz", default="data/cache/sciplex_molformer_emb.npz")
    ap.add_argument("--ridge_csv", default="../results/ridge_baseline_clean_v2.csv")
    ap.add_argument("--out", default="results/baselines_comparison.csv")
    ap.add_argument("--k_knn", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_drugs = splits["train_drugs"] + splits["val_drugs"]
    test_drugs = splits["test_drugs"]
    cell_lines = ["A549", "K562", "MCF7"]
    n_genes = 3000

    print(f"Fit drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}, Cell lines: {cell_lines}")

    # ---- Load pseudobulk pairs ----
    print("Loading pseudobulk pairs ...")
    pairs_train = build_pseudobulk(
        splits_dir / "sciplex_train.parquet",
        splits_dir / "sciplex_train_treated_counts.npy",
        splits_dir / "sciplex_train_control_counts.npy",
    )
    pairs_val = build_pseudobulk(
        splits_dir / "sciplex_val.parquet",
        splits_dir / "sciplex_val_treated_counts.npy",
        splits_dir / "sciplex_val_control_counts.npy",
    )
    pairs_test = build_pseudobulk(
        splits_dir / "sciplex_test.parquet",
        splits_dir / "sciplex_test_treated_counts.npy",
        splits_dir / "sciplex_test_control_counts.npy",
    )
    pairs_fit = merge_pairs(pairs_train, pairs_val)
    print(f"  Fit pairs: {len(pairs_fit)}, Test pairs: {len(pairs_test)}")

    # ---- SMILES -> Morgan FP ----
    smiles_df = pd.read_csv(args.smiles_csv)
    smiles_map = dict(zip(smiles_df["drug_id"].astype(str), smiles_df["smiles"]))
    all_drug_names = set()
    for d, _ in pairs_fit:
        all_drug_names.add(d)
    for d, _ in pairs_test:
        all_drug_names.add(d)

    fp_cache = {}
    missing = 0
    for drug in all_drug_names:
        smi = smiles_map.get(drug)
        fp_cache[drug] = morgan_fp(smi) if smi else np.zeros(1024)
        if not smi:
            missing += 1
    if missing:
        print(f"  WARNING: {missing} drugs missing SMILES")

    # ---- MolFormer ----
    mf = np.load(args.molformer_npz, allow_pickle=True)
    mf_emb = mf["tokens"]
    mf_ids = mf["drug_ids"]

    # ---- Compute baselines ----
    results = {}

    print("\n[1/5] Mean baseline ...")
    preds = baseline_mean(pairs_fit, pairs_test, cell_lines, n_genes)
    results["Mean"] = evaluate(pairs_test, preds)
    print(f"  R2_all={results['Mean']['R2_all']:.4f}  R2_DEG={results['Mean']['R2_DEG']:.4f}  Spearman={results['Mean']['Spearman_DEG']:.4f}")

    print("\n[2/5] Linear (Morgan FP only) ...")
    preds = baseline_linear(pairs_fit, pairs_test, fp_cache, cell_lines, args.alpha)
    results["Linear"] = evaluate(pairs_test, preds)
    print(f"  R2_all={results['Linear']['R2_all']:.4f}  R2_DEG={results['Linear']['R2_DEG']:.4f}  Spearman={results['Linear']['Spearman_DEG']:.4f}")

    print("\n[3/5] Linear-adjusted (Morgan FP + cell-line) ...")
    preds = baseline_linear_adjusted(pairs_fit, pairs_test, fp_cache, cell_lines, args.alpha)
    results["Linear-adj"] = evaluate(pairs_test, preds)
    print(f"  R2_all={results['Linear-adj']['R2_all']:.4f}  R2_DEG={results['Linear-adj']['R2_DEG']:.4f}  Spearman={results['Linear-adj']['Spearman_DEG']:.4f}")

    print(f"\n[4/5] kNN-MolFormer (k={args.k_knn}) ...")
    preds = baseline_knn_molformer(pairs_fit, pairs_test, mf_emb, mf_ids, cell_lines, k=args.k_knn)
    results["kNN-MolFormer"] = evaluate(pairs_test, preds)
    print(f"  R2_all={results['kNN-MolFormer']['R2_all']:.4f}  R2_DEG={results['kNN-MolFormer']['R2_DEG']:.4f}  Spearman={results['kNN-MolFormer']['Spearman_DEG']:.4f}")

    # Ridge_clean from saved CSV
    print("\n[5/5] Ridge_clean (from saved CSV) ...")
    ridge_csv = Path(args.ridge_csv)
    if ridge_csv.exists():
        ridge = pd.read_csv(ridge_csv)
        results["Ridge_clean"] = {
            "R2_all": float("nan"),  # ridge CSV only has per-pair pearson/spearman/mse
            "R2_DEG": float("nan"),
            "Spearman_DEG": float(ridge["spearman_top50"].mean()),
            "n_pairs": len(ridge),
        }
        print(f"  Spearman={results['Ridge_clean']['Spearman_DEG']:.4f}")
    else:
        print("  SKIP: ridge CSV not found")

    # ---- Print table ----
    print("\n" + "=" * 95)
    print("  BASELINES COMPARISON  (unseen-drug, sci-Plex internal test)")
    print("=" * 95)
    rows = []
    for name, m in results.items():
        rows.append({
            "Method": name,
            "R2(all)": f"{m['R2_all']:.4f}" if not np.isnan(m['R2_all']) else "N/A",
            "R2(top50 DEG)": f"{m['R2_DEG']:.4f}" if not np.isnan(m['R2_DEG']) else "N/A",
            "Spearman(top50 DEG)": f"{m['Spearman_DEG']:.4f}",
            "n_pairs": m["n_pairs"],
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")

    # Print SOTA bar: clean Linear-adjusted R2(DEG)
    ladj_r2 = results.get("Linear-adj", {}).get("R2_DEG", float("nan"))
    print(f"\nSOTA bar anchor: clean Linear-adjusted R2(top50 DEG) = {ladj_r2:.4f}")
    print(f"Target: CytoBridge R2(top50 DEG) >= {ladj_r2 + 0.03:.4f}")


if __name__ == "__main__":
    main()
