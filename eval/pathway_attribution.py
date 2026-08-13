#!/usr/bin/env python
"""G3-1: Pathway attribution faithfulness evaluation.

Quantifies how faithfully the model's internal pathway_attn weights reflect
true biological pathway enrichment (GSEA pre-ranked scores from MSigDB Hallmark).

Since gene-level overlap between sci-Plex's 3000 HVGs and MSigDB is only 8.5%,
we evaluate at the pathway-level directly:
  - Per-pair: Spearman r(model.pathway_attn, GSEA_score) across K=50 pathways
  - Per-pathway: agreement between model attn and GSEA across all test pairs
  - Top/bottom pathways by model-GSEA alignment

Output:
  - per_pair_pathway_faithfulness.csv
  - pathway_faithfulness_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from cytobridge.model import CytoBridge, CytoBridgeConfig


def load_model(ckpt_path: str, device: str = "cuda") -> CytoBridge:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hp = ckpt["hyper_parameters"]
    cfg = CytoBridgeConfig(**hp["model_cfg"])
    model = CytoBridge(cfg)
    state = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/v1/epoch19-val_loss325.2686.ckpt")
    ap.add_argument("--splits_dir", default="data/processed/sciplex/splits")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results/pathway_attribution")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    # ---- Load pathway names ----
    pathway_names_df = pd.read_csv(splits_dir / "pathway_names_computed.csv")
    pathway_names = pathway_names_df["pathway"].tolist()
    K = len(pathway_names)
    print(f"Loaded {K} pathway names")

    # ---- Load GSEA ground truth ----
    gsea_test = np.load(splits_dir / "sciplex_test_pathway_gsea.npy")  # [n_pairs_test, K]
    print(f"GSEA test: shape={gsea_test.shape}, range=[{gsea_test.min():.4f}, {gsea_test.max():.4f}]")

    # ---- Load test manifest for pair identity ----
    man_test = pd.read_parquet(splits_dir / "sciplex_test.parquet").reset_index(drop=True)

    # ---- Load model ----
    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt, args.device)
    print("Model loaded")

    # ---- Get model pathway_attn for each test pair ----
    from cytobridge.data import CytoBridgeDataset

    ds = CytoBridgeDataset(
        manifest_path=str(splits_dir / "sciplex_test.parquet"),
        cell_emb_path="data/cache/sciplex_scgpt_emb.npy",
        drug_emb_path="data/cache/sciplex_molformer_emb.npz",
        treated_counts_path=str(splits_dir / "sciplex_test_treated_counts.npy"),
        pathway_gsea_path=str(splits_dir / "sciplex_test_pathway_gsea.npy"),
        control_counts_path=str(splits_dir / "sciplex_test_control_counts.npy"),
        n_hard_same_drug=0, n_hard_same_cell=0,
    )

    # Group by (drug, cell_line) for pseudobulk
    pair_groups = man_test.groupby(["drug_id", "cell_line"])
    pair_keys = list(pair_groups.groups.keys())
    print(f"Test pairs: {len(pair_keys)}")

    per_pair_results = []
    for pair_idx, (drug, cl) in enumerate(pair_keys):
        idx_list = list(pair_groups.groups[(drug, cl)])
        take = np.asarray(idx_list, dtype=int)

        # Collect per-cell pathway_attn
        attns = []
        for idx in take:
            item = ds._get_one(int(idx))
            with torch.no_grad():
                out = model(
                    cell_tokens=item["cell_tokens"].unsqueeze(0).to(args.device),
                    drug_tokens=item["drug_tokens"].unsqueeze(0).to(args.device),
                    drug_mask=item["drug_mask"].unsqueeze(0).to(args.device),
                    control_counts=item.get("control_counts").unsqueeze(0).to(args.device),
                )
            pw_attn = out.get("pathway_attn")
            if pw_attn is not None:
                attns.append(pw_attn.squeeze(0).cpu().numpy())

        if attns:
            pb_attn = np.mean(attns, axis=0)  # [K] pseudobulk pathway attention
        else:
            pb_attn = np.ones(K) / K  # uniform fallback

        # GSEA ground truth for this pair (pseudobulk mean)
        gsea_pair = gsea_test[take].mean(axis=0)  # [K]

        # Spearman correlation between model attn and GSEA
        if np.std(pb_attn) > 1e-8 and np.std(gsea_pair) > 1e-8:
            sr, sp = spearmanr(pb_attn, gsea_pair)
        else:
            sr, sp = 0.0, 1.0

        per_pair_results.append({
            "drug": str(drug),
            "cell_line": str(cl),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "n_cells": len(take),
        })

    # ---- Aggregate ----
    pair_df = pd.DataFrame(per_pair_results)

    print("\n=== PATHWAY ATTRIBUTION FAITHFULNESS ===")
    print("Model: pathway_attn vs GSEA pre-ranked scores")
    print(f"Metric: per-pair Spearman r across {K} pathways")
    print()

    mean_r = pair_df["spearman_r"].mean()
    median_r = pair_df["spearman_r"].median()
    n_pos = (pair_df["spearman_r"] > 0).sum()
    n_sig = (pair_df["spearman_p"] < 0.05).sum()

    print(f"  Mean Spearman r:   {mean_r:.4f}")
    print(f"  Median Spearman r: {median_r:.4f}")
    print(f"  Pairs with r > 0:  {n_pos}/{len(pair_df)}")
    print(f"  Pairs with p<0.05: {n_sig}/{len(pair_df)}")

    # Per-pair breakdown
    print(f"\n{'Drug':35s} {'Cell':6s} {'Spearman r':>11s} {'p-value':>9s} {'Cells':>6s}")
    print("-" * 72)
    for _, row in pair_df.sort_values("spearman_r", ascending=False).iterrows():
        print(f"{row['drug']:35s} {row['cell_line']:6s} {row['spearman_r']:11.4f} {row['spearman_p']:9.4f} {row['n_cells']:6d}")

    # ---- Per-pathway agreement: model attn vs GSEA across pairs ----
    print("\n=== PER-PATHWAY MODEL-GSEA AGREEMENT ===")
    per_pw = []
    for k in range(K):
        attn_k = []
        gsea_k = []
        for pair_idx, (drug, cl) in enumerate(pair_keys):
            take = np.asarray(list(pair_groups.groups[(drug, cl)]), dtype=int)
            # Mean attn for pathway k
            attns_k = []
            for idx in take:
                item = ds._get_one(int(idx))
                with torch.no_grad():
                    out = model(
                        cell_tokens=item["cell_tokens"].unsqueeze(0).to(args.device),
                        drug_tokens=item["drug_tokens"].unsqueeze(0).to(args.device),
                        drug_mask=item["drug_mask"].unsqueeze(0).to(args.device),
                        control_counts=item.get("control_counts").unsqueeze(0).to(args.device),
                    )
                pw_a = out.get("pathway_attn")
                if pw_a is not None:
                    attns_k.append(float(pw_a.squeeze(0)[k].cpu()))
            gsea_k.append(float(gsea_test[take, k].mean()))
            attn_k.append(np.mean(attns_k) if attns_k else 1.0 / K)

        if np.std(attn_k) > 1e-8 and np.std(gsea_k) > 1e-8:
            sr, sp = spearmanr(attn_k, gsea_k)
        else:
            sr, sp = 0.0, 1.0
        per_pw.append({
            "pathway": pathway_names[k],
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "mean_attn": float(np.mean(attn_k)),
            "mean_gsea": float(np.mean(gsea_k)),
        })

    pw_df = pd.DataFrame(per_pw).sort_values("spearman_r", ascending=False)
    print(f"{'Pathway':35s} {'Spearman r':>10s} {'p-value':>9s} {'Mean Attn':>10s} {'Mean GSEA':>10s}")
    print("-" * 78)
    for _, row in pw_df.iterrows():
        print(f"{row['pathway']:35s} {row['spearman_r']:10.4f} {row['spearman_p']:9.4f} {row['mean_attn']:10.4f} {row['mean_gsea']:10.4f}")

    n_good_pw = (pw_df["spearman_r"] > 0.1).sum()
    print(f"\n  Pathways with r > 0.1: {n_good_pw}/{K}")
    print(f"  Top-5 most faithful: {list(pw_df.head(5)['pathway'])}")
    print(f"  Bottom-5 least faithful: {list(pw_df.tail(5)['pathway'])}")

    # ---- Also compute: pathway_precision_at_k using GSEA top-k ----
    from eval.metrics import pathway_precision_at_k
    precisions = []
    for pair_idx, (drug, cl) in enumerate(pair_keys):
        take = np.asarray(list(pair_groups.groups[(drug, cl)]), dtype=int)
        attns_all = []
        for idx in take:
            item = ds._get_one(int(idx))
            with torch.no_grad():
                out = model(
                    cell_tokens=item["cell_tokens"].unsqueeze(0).to(args.device),
                    drug_tokens=item["drug_tokens"].unsqueeze(0).to(args.device),
                    drug_mask=item["drug_mask"].unsqueeze(0).to(args.device),
                    control_counts=item.get("control_counts").unsqueeze(0).to(args.device),
                )
            pw_a = out.get("pathway_attn")
            if pw_a is not None:
                attns_all.append(pw_a.squeeze(0).cpu().numpy())
        pb_attn = np.mean(attns_all, axis=0) if attns_all else np.ones(K) / K
        gsea_pb = gsea_test[take].mean(axis=0)

        for k_val in [5, 10, 20]:
            p_at_k = pathway_precision_at_k(pb_attn, gsea_pb, k_val)
            precisions.append({"pair_idx": pair_idx, "drug": str(drug), "cell_line": str(cl),
                               "k": k_val, "precision": float(p_at_k)})

    prec_df = pd.DataFrame(precisions)
    print("\n=== PATHWAY PRECISION@K ===")
    for k_val in [5, 10, 20]:
        mean_p = prec_df[prec_df["k"] == k_val]["precision"].mean()
        print(f"  P@{k_val}: {mean_p:.4f}")

    # ---- Save ----
    pair_df.to_csv(out_dir / "per_pair_pathway_faithfulness.csv", index=False)
    pw_df.to_csv(out_dir / "per_pathway_agreement.csv", index=False)
    prec_df.to_csv(out_dir / "pathway_precision_at_k.csv", index=False)

    summary = {
        "task": "G3-1 pathway attribution faithfulness",
        "metric": "Spearman r(model.pathway_attn, GSEA) across K=50 pathways",
        "mean_spearman_r": float(mean_r),
        "median_spearman_r": float(median_r),
        "n_pairs_r_gt_0": int(n_pos),
        "n_pairs_total": len(pair_df),
        "n_pathways_r_gt_0.1": int(n_good_pw),
        "n_pathways_total": K,
        "precision_at_5": float(prec_df[prec_df["k"] == 5]["precision"].mean()),
        "precision_at_10": float(prec_df[prec_df["k"] == 10]["precision"].mean()),
        "precision_at_20": float(prec_df[prec_df["k"] == 20]["precision"].mean()),
    }
    with open(out_dir / "faithfulness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    return summary


if __name__ == "__main__":
    main()
