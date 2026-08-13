#!/usr/bin/env python
"""Pathway attribution: per-(drug, cell_line) pathway prediction vs GSEA truth.

Outputs:
  results/pathway_attribution/
    pred.npy   — (n_samples, K) sigmoid pathway predictions
    true.npy   — (n_samples, K) GSEA ground truth
    meta.csv   — drug_id, cell_line per sample
    per_pair.csv — per-(drug, cell_line) pathway Pearson r
    summary.json — overall stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytobridge.data import CytoBridgeDataset, collate_with_hard_negs
from cytobridge.model import CytoBridge, CytoBridgeConfig

REPO = Path(__file__).resolve().parents[1]


def pearson_r(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    xm = x - x.mean(axis=axis, keepdims=True)
    ym = y - y.mean(axis=axis, keepdims=True)
    num = (xm * ym).sum(axis=axis)
    den = np.sqrt((xm * xm).sum(axis=axis) * (ym * ym).sum(axis=axis))
    den = np.where(den < 1e-12, 1.0, den)
    return num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default="results/pathway_attribution")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    hp = ckpt.get("hyper_parameters", {})
    model_cfg_dict = hp.get("model_cfg", hp.get("model", {}))
    if isinstance(model_cfg_dict, dict) and "d_cell_in" in model_cfg_dict:
        cfg = CytoBridgeConfig(**model_cfg_dict)
    else:
        sd = ckpt.get("state_dict", ckpt)
        cfg = CytoBridgeConfig()
        cfg.d_cell_in = sd.get("backbone.cell_proj.weight", torch.empty(512, 1)).shape[1]
        cfg.d_drug_in = sd.get("backbone.drug_proj.weight", torch.empty(768, 1)).shape[1]
        cfg.d = sd.get("backbone.cell_proj.weight", torch.empty(256, 1)).shape[0]
        n_layers = 0
        for k in sd:
            if k.startswith("backbone.layers."):
                n_layers = max(n_layers, int(k.split(".")[2]) + 1)
        if n_layers:
            cfg.n_layers = n_layers
        cfg.n_genes = sd.get("zinb.mu.weight", torch.empty(3000, 1)).shape[0]
        cfg.K_pathways = sd.get("pathway_head.fc.weight", torch.empty(50, 1)).shape[0]
        cfg.contrastive_dim = sd.get("contrast.fc.weight", torch.empty(128, 1)).shape[0]

    K = cfg.K_pathways
    print(f"Model: d_cell={cfg.d_cell_in}, d_drug={cfg.d_drug_in}, d={cfg.d}, "
          f"n_layers={cfg.n_layers}, n_genes={cfg.n_genes}, K={K}")

    model = CytoBridge(cfg)
    state = ckpt.get("state_dict", ckpt)
    stripped = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(stripped, strict=False)
    model = model.to(device)
    model.eval()

    # --- Load test data ---
    BASE = REPO / "data" / "processed" / "sciplex" / "splits"
    CACHE = REPO / "data" / "cache"
    print("Loading test dataset...")
    ds = CytoBridgeDataset(
        manifest_path=BASE / "sciplex_test.parquet",
        cell_emb_path=CACHE / "sciplex_scgpt_emb.npy",
        drug_emb_path=CACHE / "sciplex_molformer_emb.npz",
        treated_counts_path=BASE / "sciplex_test_treated_counts.npy",
        pathway_gsea_path=BASE / "sciplex_test_pathway_gsea.npy",
        control_counts_path=BASE / "sciplex_test_control_counts.npy",
        n_hard_same_drug=0, n_hard_same_cell=0, seed=42,
    )
    print(f"Test samples: {len(ds)}")

    # Load pathway names if available
    pathway_names_path = BASE / "pathway_names_computed.csv"
    pathway_names = None
    if pathway_names_path.exists():
        pathway_names = pd.read_csv(pathway_names_path)
        print(f"Pathway names: {len(pathway_names)} entries")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_with_hard_negs)

    # --- Run inference ---
    all_pathway_pred = []
    all_pathway_gt = []
    all_drug_ids = []
    all_cell_lines = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            out = model(batch_dev["cell_tokens"], batch_dev["drug_tokens"],
                        batch_dev["drug_mask"])

            all_pathway_pred.append(torch.sigmoid(out["pathway_pred"]).cpu().numpy())
            all_pathway_gt.append(batch_dev["pathway_gsea"].cpu().numpy())
            all_drug_ids.extend(batch_dev["drug_ids"])
            all_cell_lines.extend(batch_dev["cell_lines"])

    path_pred = np.concatenate(all_pathway_pred, axis=0)
    path_gt = np.concatenate(all_pathway_gt, axis=0)
    drug_ids = all_drug_ids
    cell_lines = all_cell_lines

    # --- Save raw arrays ---
    np.save(out_dir / "pred.npy", path_pred)
    np.save(out_dir / "true.npy", path_gt)
    print(f"Saved pred.npy {path_pred.shape}, true.npy {path_gt.shape}")

    meta = pd.DataFrame({"drug_id": drug_ids, "cell_line": cell_lines})
    meta.to_csv(out_dir / "meta.csv", index=False)
    print(f"Saved meta.csv ({len(meta)} rows)")

    # --- Overall pathway r ---
    path_r = pearson_r(path_gt, path_pred)
    mask = ~np.isnan(path_r)
    overall_r_mean = float(np.mean(path_r[mask])) if mask.sum() > 0 else float("nan")
    overall_r_median = float(np.median(path_r[mask])) if mask.sum() > 0 else float("nan")
    print(f"\nOverall pathway r: mean={overall_r_mean:.4f}, median={overall_r_median:.4f}")

    # --- Per-(drug, cell_line) pathway r ---
    df = pd.DataFrame({
        "drug_id": drug_ids,
        "cell_line": cell_lines,
        "path_pred": list(path_pred),
        "path_gt": list(path_gt),
    })

    per_pair_rows = []
    for (drug, cl), grp in df.groupby(["drug_id", "cell_line"]):
        pred_pb = np.mean(np.stack(grp["path_pred"].values), axis=0)  # (K,)
        gt_pb = np.mean(np.stack(grp["path_gt"].values), axis=0)      # (K,)
        r = pearson_r(gt_pb, pred_pb, axis=0)
        per_pair_rows.append({
            "drug": drug,
            "cell_line": cl,
            "n_samples": len(grp),
            "pathway_pearson_r": float(np.clip(r, -1, 1)),
        })

    per_pair_df = pd.DataFrame(per_pair_rows)
    per_pair_df.to_csv(out_dir / "per_pair.csv", index=False)
    print(f"Saved per_pair.csv ({len(per_pair_df)} rows)")

    pairwise_r_mean = per_pair_df.pathway_pearson_r.mean()
    pairwise_r_median = per_pair_df.pathway_pearson_r.median()
    print(f"Per-pair pathway r: mean={pairwise_r_mean:.4f}, median={pairwise_r_median:.4f}")

    # --- Summary ---
    summary = {
        "ckpt": args.ckpt,
        "K_pathways": K,
        "n_test_samples": len(ds),
        "overall_pathway_r_mean": overall_r_mean,
        "overall_pathway_r_median": overall_r_median,
        "pairwise_pathway_r_mean": pairwise_r_mean,
        "pairwise_pathway_r_median": pairwise_r_median,
        "n_pairs": len(per_pair_df),
        "per_pair_summary": per_pair_df.pathway_pearson_r.describe().to_dict(),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Saved summary.json")

    # --- Per-pair detail ---
    print("\n" + "=" * 60)
    print("Per-(drug, cell_line) Pathway Pearson r")
    print("=" * 60)
    print(per_pair_df.to_string(index=False))

    # Check if any pairs have strong negative correlation (sanity check)
    n_negative = (per_pair_df.pathway_pearson_r < 0).sum()
    print(f"\nPairs with negative r: {n_negative}/{len(per_pair_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
