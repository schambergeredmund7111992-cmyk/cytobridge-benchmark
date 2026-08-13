"""
eval/baselines/scgpt_zeroshot.py
--------------------------------
scGPT zero-shot perturbation prediction baseline.

Uses cached scGPT embeddings without fine-tuning. For each test pair, it copies
the logFC profile from the nearest training cell state in scGPT embedding space.

This baseline is essential — it shows what scGPT can do "out of the box"
on our task. Our CytoBridge must outperform this.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


def mean_pool_tokens(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=1)
    return arr


def pair_logfc(adata: sc.AnnData, drug_col: str = "drug", cell_col: str = "cell_line",
               control_label: str = "DMSO") -> tuple[pd.DataFrame, np.ndarray]:
    rows, values = [], []
    for (drug, cl), idx in adata.obs.groupby([drug_col, cell_col]).groups.items():
        if drug == control_label:
            continue
        ctrl_idx = adata.obs.index[(adata.obs[drug_col] == control_label) & (adata.obs[cell_col] == cl)]
        if len(ctrl_idx) == 0:
            continue
        treated = np.asarray(adata[idx].X.mean(axis=0)).reshape(-1)
        control = np.asarray(adata[ctrl_idx].X.mean(axis=0)).reshape(-1)
        values.append(np.log1p(treated) - np.log1p(control))
        rows.append({"drug": drug, "cell_line": cl, "cell_indices": list(adata.obs.index.get_indexer(idx))})
    return pd.DataFrame(rows), np.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path,
                        default=Path("~/.cache/scgpt/whole_human").expanduser())
    parser.add_argument("--test_h5ad", type=Path, required=True)
    parser.add_argument("--train_h5ad", type=Path, required=True)
    parser.add_argument("--train_cell_emb", type=Path, required=True)
    parser.add_argument("--test_cell_emb", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/scgpt_zeroshot.csv"))
    args = parser.parse_args()

    train = sc.read_h5ad(args.train_h5ad)
    test = sc.read_h5ad(args.test_h5ad)
    train_emb = mean_pool_tokens(np.load(args.train_cell_emb, mmap_mode="r"))
    test_emb = mean_pool_tokens(np.load(args.test_cell_emb, mmap_mode="r"))
    train_meta, train_logfc = pair_logfc(train)
    test_meta, test_logfc = pair_logfc(test)

    train_pair_emb = np.stack([
        train_emb[np.asarray(row["cell_indices"], dtype=int)].mean(axis=0)
        for _, row in train_meta.iterrows()
    ])
    test_pair_emb = np.stack([
        test_emb[np.asarray(row["cell_indices"], dtype=int)].mean(axis=0)
        for _, row in test_meta.iterrows()
    ])
    train_pair_emb = train_pair_emb / np.maximum(np.linalg.norm(train_pair_emb, axis=1, keepdims=True), 1e-8)
    test_pair_emb = test_pair_emb / np.maximum(np.linalg.norm(test_pair_emb, axis=1, keepdims=True), 1e-8)
    nn_idx = np.argmax(test_pair_emb @ train_pair_emb.T, axis=1)
    pred = train_logfc[nn_idx]

    rows = []
    for i, (_, m) in enumerate(test_meta.iterrows()):
        true = test_logfc[i]
        p = pred[i]
        top = np.argsort(-np.abs(true))[:50]
        rows.append({
            "drug": m["drug"],
            "cell_line": m["cell_line"],
            "pearson_top50": stats.pearsonr(true[top], p[top]).statistic,
            "spearman_top50": stats.spearmanr(true[top], p[top]).statistic,
            "nearest_train_drug": train_meta.iloc[int(nn_idx[i])]["drug"],
            "nearest_train_cell_line": train_meta.iloc[int(nn_idx[i])]["cell_line"],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
