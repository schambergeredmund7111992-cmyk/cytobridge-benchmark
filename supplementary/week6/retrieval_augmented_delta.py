#!/usr/bin/env python
"""T1-2: Retrieval-Augmented Delta.
====================================
In drug embedding (MolFormer) space, k=5 nearest training drugs' pseudobulk
deltas are similarity-weighted-averaged as a PRIOR. A lightweight per-gene
Ridge correction head g_theta is then stacked on top.

Ablation:
  (A) Pure retrieval interpolation (kNN-weighted average, k=5)
  (B) Retrieval + correction head (per-gene Ridge on [retrieval_prior, cell_line])

Benchmark: clean Linear-adjusted (Ridge on Morgan FP + cell-line one-hot).

Reports: R2(DEG) + Spearman(top50 DEG) on sci-Plex internal test split.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Helpers (mirror all_baselines.py for consistency)
# ---------------------------------------------------------------------------
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def build_pseudobulk(manifest_path, treated_path, control_path):
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
    out = {}
    for d in dicts:
        out.update(d)
    return out


def evaluate(pairs_dict, preds_dict, top_k=50):
    """Per-gene R2(all), per-gene R2(top-K DEG), per-pair Spearman(top-K DEG)."""
    keys = list(pairs_dict.keys())
    n_genes = len(pairs_dict[keys[0]]["log_fc"])

    true_mat = np.stack([pairs_dict[k]["log_fc"] for k in keys])
    pred_mat = np.stack([preds_dict.get(k, np.zeros(n_genes)) for k in keys])

    # Per-gene R2 (all genes)
    ss_res = ((true_mat - pred_mat) ** 2).sum(axis=0)
    ss_tot = ((true_mat - true_mat.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_gene = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0)
    r2_all = float(r2_per_gene.mean())

    # Per-pair Spearman on top-K DEG
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
    for i in range(len(true_mat)):
        deg_union.update(np.argsort(-np.abs(true_mat[i]))[:top_k].tolist())
    deg_genes = sorted(deg_union)
    r2_deg = float(r2_per_gene[deg_genes].mean()) if deg_genes else float("nan")

    return {
        "R2_all": r2_all,
        "R2_DEG": r2_deg,
        "Spearman_DEG": float(np.mean(spear_list)) if spear_list else float("nan"),
        "n_pairs": len(keys),
        "n_deg_genes": len(deg_genes),
    }


# ---------------------------------------------------------------------------
# T1-2 Core: Retrieval-Augmented Delta
# ---------------------------------------------------------------------------
def retrieval_prior(
    fit_pairs, query_pairs, molformer_emb, molformer_ids,
    cell_lines, k=5, temperature=5.0, leave_out_query_drugs=True,
):
    """(A) Pure retrieval: kNN-weighted average in MolFormer space.

    For each query drug, finds k nearest fit drugs by cosine similarity,
    then for each cell line averages their per-cell-line pseudobulk delta
    weighted by softmax(similarity * temperature).

    When leave_out_query_drugs=True and a query drug exists in the fit set,
    it is excluded from candidates (LOO) to prevent the prior from being
    self-dominated. For genuinely unseen test drugs this is a no-op.
    """
    n_genes = 3000
    emb = molformer_emb.mean(axis=1)  # [N_drugs, 768] mean pool over tokens
    id_to_vec = {}
    for i, d in enumerate(molformer_ids):
        v = emb[i]
        id_to_vec[str(d)] = v / (np.linalg.norm(v) + 1e-12)

    # Per-drug per-cell-line delta from fit pairs
    fit_drugs = sorted({d for d, _ in fit_pairs})
    fit_drug_to_idx = {d: i for i, d in enumerate(fit_drugs)}
    fit_delta = {}
    for drug in fit_drugs:
        fit_delta[drug] = {}
        for cl in cell_lines:
            key = (drug, cl)
            fit_delta[drug][cl] = fit_pairs[key]["log_fc"] if key in fit_pairs else np.zeros(n_genes)

    fit_vecs = np.stack([id_to_vec.get(d, np.zeros(768)) for d in fit_drugs])

    query_drugs = sorted({d for d, _ in query_pairs})
    preds = {}
    for drug in query_drugs:
        q = id_to_vec.get(drug)
        if q is None:
            for cl in cell_lines:
                preds[(drug, cl)] = np.zeros(n_genes)
            continue
        sim = fit_vecs @ q

        # Leave-out-self: if query drug is in the fit set, exclude it
        if leave_out_query_drugs and drug in fit_drug_to_idx:
            loo_idx = fit_drug_to_idx[drug]
            sim = sim.copy()
            sim[loo_idx] = -np.inf

        top_k = np.argsort(-sim)[:k]
        w = np.exp(sim[top_k] * temperature)
        w /= w.sum()

        for cl in cell_lines:
            delta = np.zeros(n_genes)
            for i, weight in zip(top_k, w):
                delta += weight * fit_delta[fit_drugs[i]].get(cl, np.zeros(n_genes))
            preds[(drug, cl)] = delta
    return preds


def retrieval_plus_correction(
    fit_pairs, test_pairs, retrieval_prior_fit, retrieval_prior_test,
    cell_lines, alpha=1.0,
):
    """(B) Retrieval + per-gene Ridge correction head g_theta.

    For each gene g, fit:
        true_delta[drug, cl][g] ~ retrieval_prior[drug, cl][g] + one_hot(cl)

    retrieval_prior_fit should be LOO-computed for fit drugs (no self-leak).
    retrieval_prior_test is used for test-time prediction.

    This learns gene-specific scaling of the retrieval prior plus
    cell-line-specific offsets.
    """
    n_genes = 3000
    fit_keys = list(fit_pairs)

    # Build feature matrix: [retrieval_prior_value, A549, K562, MCF7]
    cell_enc = OneHotEncoder(sparse_output=False).fit(
        np.array(cell_lines).reshape(-1, 1)
    )
    retrieval_vals = np.array([retrieval_prior_fit[k] for k in fit_keys])  # [N, G]
    cell_feat = cell_enc.transform(
        np.array([cl for _, cl in fit_keys]).reshape(-1, 1)
    )  # [N, 3]

    # Per-gene Ridge
    models, scaler_list = [], []
    for g in range(n_genes):
        X_g = np.column_stack([retrieval_vals[:, g], cell_feat])  # [N, 4]
        y_g = np.array([fit_pairs[k]["log_fc"][g] for k in fit_keys])
        scaler = StandardScaler().fit(X_g)
        X_s = scaler.transform(X_g)
        if np.std(y_g) < 1e-8:
            models.append(("const", float(np.mean(y_g))))
            scaler_list.append(scaler)
        else:
            m = Ridge(alpha=alpha, fit_intercept=True, max_iter=5000)
            m.fit(X_s, y_g)
            models.append(("ridge", m))
            scaler_list.append(scaler)

    # Predict for test pairs
    test_keys = list(test_pairs)
    test_retrieval = np.array([retrieval_prior_test.get(k, np.zeros(n_genes))
                                for k in test_keys])
    test_cell = cell_enc.transform(
        np.array([cl for _, cl in test_keys]).reshape(-1, 1)
    )
    preds = {}
    for i, k in enumerate(test_keys):
        pred_g = np.zeros(n_genes)
        for g in range(n_genes):
            X_g = np.concatenate([[test_retrieval[i, g]], test_cell[i]])
            X_s = scaler_list[g].transform(X_g.reshape(1, -1))
            if models[g][0] == "const":
                pred_g[g] = models[g][1]
            else:
                pred_g[g] = float(models[g][1].predict(X_s)[0])
        preds[k] = pred_g
    return preds


# ---------------------------------------------------------------------------
# Linear-adjusted baseline (reproduced for self-contained comparison)
# ---------------------------------------------------------------------------
def baseline_linear_adjusted(fit_pairs, test_pairs, fp_cache, cell_lines, alpha=1.0):
    fit_keys = list(fit_pairs)
    cell_enc = OneHotEncoder(sparse_output=False).fit(np.array(cell_lines).reshape(-1, 1))

    train_fp = np.stack([fp_cache[d] for d, _ in fit_keys])
    train_cell = cell_enc.transform(np.array([c for _, c in fit_keys]).reshape(-1, 1))
    train_X = np.hstack([train_fp, train_cell])
    train_Y = np.stack([fit_pairs[k]["log_fc"] for k in fit_keys])

    scaler = StandardScaler().fit(train_X)
    X_s = scaler.transform(train_X)
    models = []
    for g in range(train_Y.shape[1]):
        y = train_Y[:, g]
        if np.std(y) < 1e-8:
            models.append(("const", float(np.mean(y))))
        else:
            m = Ridge(alpha=alpha, fit_intercept=True, max_iter=5000)
            m.fit(X_s, y)
            models.append(("ridge", m))

    test_keys = list(test_pairs)
    test_fp = np.stack([fp_cache[d] for d, _ in test_keys])
    test_cell = cell_enc.transform(np.array([c for _, c in test_keys]).reshape(-1, 1))
    test_X = np.hstack([test_fp, test_cell])
    X_s_test = scaler.transform(test_X)

    preds = {}
    for i, k in enumerate(test_keys):
        pred = np.zeros(train_Y.shape[1])
        for g, (kind, m) in enumerate(models):
            pred[g] = m if kind == "const" else m.predict(X_s_test[i:i+1])[0]
        preds[k] = pred
    return preds


# ---------------------------------------------------------------------------
# Morgan fingerprint helper
# ---------------------------------------------------------------------------
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits)
    for i, bit in enumerate(fp):
        if bit:
            arr[i] = 1.0
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="T1-2: Retrieval-Augmented Delta")
    ap.add_argument("--splits_dir", default="data/processed/sciplex/splits")
    ap.add_argument("--splits_json", default="data/processed/sciplex/splits/internal_splits.json")
    ap.add_argument("--smiles_csv", default="data/processed/sciplex/drugs_canonical.csv")
    ap.add_argument("--molformer_npz", default="data/cache/sciplex_molformer_emb.npz")
    ap.add_argument("--out", default="results/t1_2_retrieval_augmented_delta.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge alpha for correction head (and Linear-adjusted baseline)")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_drugs = splits["train_drugs"] + splits["val_drugs"]
    test_drugs = splits["test_drugs"]
    cell_lines = ["A549", "K562", "MCF7"]
    n_genes = 3000

    print(f"T1-2 Retrieval-Augmented Delta")
    print(f"  k={args.k}, temperature={args.temperature}, alpha={args.alpha}")
    print(f"  Fit drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}")
    print(f"  Cell lines: {cell_lines}")

    # ---- Load pseudobulk pairs ----
    print("\nLoading pseudobulk pairs ...")
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

    # ---- SMILES -> Morgan FP (for Linear-adjusted baseline) ----
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

    # ---- (A) Pure retrieval interpolation (on UNSEEN test drugs, no LOO needed) ----
    print(f"\n[A] Pure retrieval interpolation (kNN-MolFormer, k={args.k}, T={args.temperature}) ...")
    preds_retrieval_test = retrieval_prior(
        pairs_fit, pairs_test, mf_emb, mf_ids, cell_lines,
        k=args.k, temperature=args.temperature, leave_out_query_drugs=False,
    )
    res_retrieval = evaluate(pairs_test, preds_retrieval_test)
    print(f"  R2_all={res_retrieval['R2_all']:.4f}  "
          f"R2_DEG={res_retrieval['R2_DEG']:.4f}  "
          f"Spearman={res_retrieval['Spearman_DEG']:.4f}")

    # ---- (B) Retrieval + correction head ----
    # Correction head is trained on LOO retrieval priors for FIT drugs to avoid
    # self-domination (kNN of a fit drug would otherwise include itself).
    print(f"\n[B] Retrieval + per-gene Ridge correction head (alpha={args.alpha}) ...")
    print("    Computing LOO retrieval prior for fit pairs (correction head training)...")
    preds_retrieval_fit_loo = retrieval_prior(
        pairs_fit, pairs_fit, mf_emb, mf_ids, cell_lines,
        k=args.k, temperature=args.temperature, leave_out_query_drugs=True,
    )
    preds_corrected = retrieval_plus_correction(
        pairs_fit, pairs_test, preds_retrieval_fit_loo, preds_retrieval_test,
        cell_lines, alpha=args.alpha,
    )
    res_corrected = evaluate(pairs_test, preds_corrected)
    print(f"  R2_all={res_corrected['R2_all']:.4f}  "
          f"R2_DEG={res_corrected['R2_DEG']:.4f}  "
          f"Spearman={res_corrected['Spearman_DEG']:.4f}")

    # ---- (C) Linear-adjusted baseline (SOTA bar) ----
    print(f"\n[C] Linear-adjusted baseline (Morgan FP + cell-line Ridge, alpha={args.alpha}) ...")
    preds_linear_adj = baseline_linear_adjusted(
        pairs_fit, pairs_test, fp_cache, cell_lines, alpha=args.alpha,
    )
    res_linear_adj = evaluate(pairs_test, preds_linear_adj)
    print(f"  R2_all={res_linear_adj['R2_all']:.4f}  "
          f"R2_DEG={res_linear_adj['R2_DEG']:.4f}  "
          f"Spearman={res_linear_adj['Spearman_DEG']:.4f}")

    # ---- Report ----
    print("\n" + "=" * 95)
    print("  T1-2  RETRIEVAL-AUGMENTED DELTA  (unseen-drug, sci-Plex internal test)")
    print("=" * 95)
    rows = []
    for name, m in [
        ("(A) Pure retrieval (kNN)", res_retrieval),
        ("(B) Retrieval + g_theta correction", res_corrected),
        ("(C) Linear-adjusted (SOTA bar)", res_linear_adj),
    ]:
        rows.append({
            "Method": name,
            "R2(all)": f"{m['R2_all']:.4f}" if not np.isnan(m['R2_all']) else "N/A",
            "R2(top50 DEG)": f"{m['R2_DEG']:.4f}" if not np.isnan(m['R2_DEG']) else "N/A",
            "Spearman(top50 DEG)": f"{m['Spearman_DEG']:.4f}",
            "n_pairs": m["n_pairs"],
            "n_deg_genes": m["n_deg_genes"],
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    # Compute deltas
    delta_retrieval_vs_linear = res_retrieval["Spearman_DEG"] - res_linear_adj["Spearman_DEG"]
    delta_corrected_vs_linear = res_corrected["Spearman_DEG"] - res_linear_adj["Spearman_DEG"]
    delta_correction_gain = res_corrected["Spearman_DEG"] - res_retrieval["Spearman_DEG"]
    print(f"\n  Δ Spearman (retrieval vs Linear-adj): {delta_retrieval_vs_linear:+.4f}")
    print(f"  Δ Spearman (retrieval+correction vs Linear-adj): {delta_corrected_vs_linear:+.4f}")
    print(f"  Δ Spearman (correction head gain): {delta_correction_gain:+.4f}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")

    # Save detailed per-pair predictions for downstream analysis
    detail_out = Path(args.out).with_suffix(".details.csv")
    detail_rows = []
    for k in pairs_test:
        detail_rows.append({
            "drug": k[0], "cell_line": k[1],
            "true_logfc_norm": float(np.linalg.norm(pairs_test[k]["log_fc"])),
            "retrieval_prior": preds_retrieval_test.get(k, np.zeros(n_genes)).tolist(),
            "retrieval_corrected": preds_corrected.get(k, np.zeros(n_genes)).tolist(),
            "linear_adj": preds_linear_adj.get(k, np.zeros(n_genes)).tolist(),
            "true": pairs_test[k]["log_fc"].tolist(),
        })
    pd.DataFrame(detail_rows).to_csv(detail_out, index=False)
    print(f"Per-pair details saved to {detail_out}")


if __name__ == "__main__":
    main()
