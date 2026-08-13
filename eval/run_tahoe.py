"""
eval/run_tahoe.py
-----------------
Evaluate CytoBridge on Tahoe-100M two-tier external splits:
    --split external_1: unseen drugs (drug not in sci-Plex training set)
    --split external_2: unseen cell lines

The third tier (both unseen) was dropped — too few samples for confident CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.run_internal import aggregate_by_pair, load_model, predict_all
from eval.metrics import bootstrap_ci, per_pair_pearson, per_pair_spearman
from cytobridge.data import CytoBridgeDataset


def _assert_cache_aligned(split: str, manifest_path: Path, cache_dir: Path,
                          treated_counts_path: Path, control_counts_path: Path,
                          gsea_path: Path, cell_emb_path: Path) -> None:
    """Refuse to evaluate if the caches don't match the manifest.

    Tahoe caches must be produced by `data/build_tahoe_cache.py`. That script
    writes a `<split>_cache_meta.json` with shapes + manifest SHA256 +
    per-array SHA256 so we can detect any drift before the loader silently
    indexes a misaligned array.
    """
    meta_path = cache_dir / f"tahoe_{split}_cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run "
            f"`python data/build_tahoe_cache.py --split {split}` "
            "before evaluation; reading caches without alignment metadata "
            "is a leakage risk."
        )
    meta = json.loads(meta_path.read_text())

    h = hashlib.sha256()
    with open(manifest_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta.get("manifest_sha256"):
        raise ValueError(
            f"Manifest {manifest_path} has changed since the cache was built "
            f"(sha256 mismatch vs {meta_path}). Rebuild the cache."
        )

    expected_manifest_rows = meta["manifest_rows"]
    expected_h5ad_rows = meta["h5ad_rows"]

    treated_arr = np.load(treated_counts_path, mmap_mode="r")
    if treated_arr.shape[0] != expected_manifest_rows:
        raise ValueError(
            f"{treated_counts_path}: shape[0]={treated_arr.shape[0]}, "
            f"expected {expected_manifest_rows} (manifest rows)."
        )
    control_arr = np.load(control_counts_path, mmap_mode="r")
    if control_arr.shape[0] != expected_manifest_rows:
        raise ValueError(
            f"{control_counts_path}: shape[0]={control_arr.shape[0]}, "
            f"expected {expected_manifest_rows} (manifest rows)."
        )
    gsea_arr = np.load(gsea_path, mmap_mode="r")
    if gsea_arr.shape[0] != expected_manifest_rows:
        raise ValueError(
            f"{gsea_path}: shape[0]={gsea_arr.shape[0]}, "
            f"expected {expected_manifest_rows} (manifest rows)."
        )
    if cell_emb_path.exists():
        cell_arr = np.load(cell_emb_path, mmap_mode="r")
        if cell_arr.shape[0] != expected_h5ad_rows:
            raise ValueError(
                f"{cell_emb_path}: shape[0]={cell_arr.shape[0]}, "
                f"expected {expected_h5ad_rows} (sliced-h5ad rows)."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=["external_1", "external_2"], required=True)
    parser.add_argument("--manifest_dir", type=Path,
                        default=Path("data/processed/tahoe/splits/"))
    # build_external_split.py writes one sliced h5ad per split (with split-local
    # indices), so each split needs its own cache file. Defaults are derived
    # from --split; pass an explicit path to override.
    parser.add_argument("--cell_emb", type=Path, default=None)
    parser.add_argument("--drug_emb", type=Path, default=None)
    parser.add_argument("--counts", type=Path, default=None)
    parser.add_argument("--control_counts", type=Path, default=None)
    parser.add_argument("--gsea", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("results/"))
    args = parser.parse_args()
    raise SystemExit(
        "LEGACY TAHOE ENTRYPOINT DISABLED: external_1/external_2 scoring conflicts "
        "with protocol 1.4. Run the frozen Tahoe refit via "
        "python -m scripts.run_frozen_cytobridge and package the frozen test split "
        "with eval.predict_cytobridge/eval.package_artifact."
    )
    cache = Path("data/cache")
    if args.cell_emb is None:
        args.cell_emb = cache / f"tahoe_{args.split}_scgpt_emb.npy"
    if args.drug_emb is None:
        args.drug_emb = cache / f"tahoe_{args.split}_molformer_emb.npz"
    if args.counts is None:
        args.counts = cache / f"tahoe_{args.split}_treated_counts.npy"
    if args.control_counts is None:
        args.control_counts = cache / f"tahoe_{args.split}_control_counts.npy"
    if args.gsea is None:
        args.gsea = cache / f"tahoe_{args.split}_pathway_gsea.npy"

    manifest_path = args.manifest_dir / f"tahoe_{args.split}.parquet"
    _assert_cache_aligned(
        split=args.split,
        manifest_path=manifest_path,
        cache_dir=cache,
        treated_counts_path=args.counts,
        control_counts_path=args.control_counts,
        gsea_path=args.gsea,
        cell_emb_path=args.cell_emb,
    )

    model = load_model(args.ckpt)
    ds = CytoBridgeDataset(
        manifest_path=manifest_path,
        cell_emb_path=args.cell_emb, drug_emb_path=args.drug_emb,
        treated_counts_path=args.counts, pathway_gsea_path=args.gsea,
        control_counts_path=args.control_counts if args.control_counts.exists() else None,
        n_hard_same_drug=0, n_hard_same_cell=0,
    )
    pred_mu, true_counts, ctrl_counts, drugs, cells = predict_all(model, ds)
    preds, trues, drugs, cells = aggregate_by_pair(
        pred_mu, true_counts, ctrl_counts, drugs, cells
    )
    pearson = per_pair_pearson(trues, preds)
    spearman = per_pair_spearman(trues, preds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"drug_id": drugs, "cell_line": cells,
                  "pearson_top50": pearson, "spearman_top50": spearman})\
        .to_csv(args.out_dir / f"cytobridge_tahoe_{args.split}.csv", index=False)

    p_mean, p_lo, p_hi = bootstrap_ci(pearson)
    s_mean, s_lo, s_hi = bootstrap_ci(spearman)
    print(f"\n=== CytoBridge Tahoe-100M {args.split} ===")
    print(f"Pearson@50:  {p_mean:.4f}  [95% CI {p_lo:.4f}, {p_hi:.4f}]")
    print(f"Spearman@50: {s_mean:.4f}  [95% CI {s_lo:.4f}, {s_hi:.4f}]")


if __name__ == "__main__":
    main()
