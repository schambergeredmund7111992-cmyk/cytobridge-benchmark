#!/usr/bin/env python3
"""Generate plate-matched control counts for arm B.

Groups treated cells by (cell_line, plate) and computes the mean
DMSO control expression per group.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate plate-matched control counts")
    parser.add_argument("--h5ad", required=True, help="Path to processed h5ad file")
    parser.add_argument("--manifest", required=True, help="Path to train manifest parquet")
    parser.add_argument("--match_field", default="plate", help="Field to match on (plate or replicate)")
    parser.add_argument("--output", required=True, help="Output path for matched control counts .npy")
    parser.add_argument("--control_counts", default=None,
                        help="Path to original control counts .npy (default: auto-detect)")
    args = parser.parse_args()

    print(f"[generate] Loading manifest: {args.manifest}")
    manifest = pd.read_parquet(args.manifest)
    n_rows = len(manifest)
    print(f"[generate] Manifest: {n_rows} rows, cols={list(manifest.columns)}")

    print(f"[generate] Loading h5ad obs: {args.h5ad}")
    import anndata
    adata = anndata.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    print(f"[generate] h5ad: {adata.n_obs} cells, obs cols={list(obs.columns)}")

    if args.match_field not in obs.columns:
        raise ValueError(
            f"match_field '{args.match_field}' not in h5ad.obs. "
            f"Available: {list(obs.columns)}"
        )

    # Map cell_idx to match_field value
    cell_idx_to_plate = {}
    for idx in range(adata.n_obs):
        cell_idx_to_plate[idx] = obs.iloc[idx][args.match_field]

    # Add match_field column to manifest using cell_idx
    manifest["_match_field"] = manifest["cell_idx"].map(cell_idx_to_plate)
    n_missing = manifest["_match_field"].isna().sum()
    if n_missing > 0:
        print(f"[generate] WARNING: {n_missing}/{n_rows} rows missing match_field")

    # Group by (cell_line, match_field)
    groups = manifest.groupby(["cell_line", "_match_field"])
    n_groups = len(groups)
    print(f"[generate] {n_groups} unique (cell_line, {args.match_field}) groups")

    # Verify coverage: count groups with treated samples
    n_with_data = sum(1 for _, idxs in groups.indices.items() if len(idxs) > 0)
    coverage = n_with_data / max(n_groups, 1) * 100
    print(f"[generate] Coverage: {n_with_data}/{n_groups} groups ({coverage:.1f}%) with treated samples")

    # Load control counts
    if args.control_counts:
        control_counts_path = args.control_counts
    else:
        manifest_dir = Path(args.manifest).parent
        control_counts_path = str(manifest_dir / "sciplex_train_control_counts.npy")

    print(f"[generate] Loading control counts: {control_counts_path}")
    control_counts = np.load(control_counts_path, mmap_mode="r")
    n_genes = control_counts.shape[1]

    matched_controls = np.zeros((n_rows, n_genes), dtype=np.float32)

    for (cl, plate_val), idxs in groups.indices.items():
        group_idxs = list(idxs)
        group_mean = control_counts[group_idxs].mean(axis=0)
        matched_controls[group_idxs] = group_mean

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), matched_controls)
    print(f"[generate] Saved to {args.output}")
    print(f"[generate] Shape: {matched_controls.shape}, dtype: {matched_controls.dtype}")

    adata.file.close()


if __name__ == "__main__":
    main()
