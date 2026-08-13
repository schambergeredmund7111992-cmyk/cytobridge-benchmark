#!/usr/bin/env python
"""Rebuild control_counts as the per-cell-line DMSO PSEUDOBULK MEAN (broadcast to
every row of each split), so that THREE things use ONE identical control:
  (a) the residual decoder baseline  mu = expm1(log1p(control)+delta)
  (b) the run_internal eval target   true_logFC = log1p(treated_pb) - log1p(ctrl_pb)
  (c) the ridge baseline             logFC = log1p(treated_pb) - log1p(all-DMSO mean)

Fixes: the old cache stored ONE randomly-sampled DMSO single cell
per treated row (preprocess.py: rng.choice(candidates)), while ridge uses the
all-DMSO cell-line mean. Different controls -> different top-50 DEG sets -> the
Δ-vs-ridge headline is not apples-to-apples. After this script every path uses
the cell-line DMSO mean, so the comparison is valid AND the residual baseline is
less noisy.

Server run (needs anndata; backs up each old npy to .bak first):
  cd code
  python scripts/rebuild_control_pseudobulk.py \
    --h5ad data/processed/sciplex/sciplex_processed.h5ad \
    --splits_dir data/processed/sciplex/splits \
    --splits train val test \
    --cell_col cell_line --pert_col drug --control_label DMSO --layer counts

Then retrain / re-eval normally; nothing else changes (configs already point at
sciplex_<split>_control_counts.npy).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure, unit-tested core (no I/O) — see tests/test_rebuild_control.py
# ---------------------------------------------------------------------------
def cellline_control_means(
    counts: np.ndarray,          # [N, G] raw counts
    cell_lines: np.ndarray,      # [N] cell-line label per cell
    is_control: np.ndarray,      # [N] bool, True for DMSO/control cells
) -> dict:
    """Per-cell-line mean expression over CONTROL cells only -> {cell_line: [G]}."""
    cl = np.asarray(cell_lines)
    ctrl = np.asarray(is_control, dtype=bool)
    out: dict = {}
    for c in np.unique(cl[ctrl]):
        m = (cl == c) & ctrl
        out[str(c)] = counts[m].mean(axis=0).astype(np.float32)
    return out


def broadcast_control(
    cl_to_mean: dict,
    manifest_cell_lines,         # [N_rows] cell-line per manifest row
    n_genes: int,
) -> tuple[np.ndarray, list]:
    """Broadcast each row's cell-line control mean -> [N_rows, G]. Returns
    (array, missing_cell_lines)."""
    rows = list(manifest_cell_lines)
    arr = np.zeros((len(rows), n_genes), dtype=np.float32)
    missing = []
    for i, c in enumerate(rows):
        v = cl_to_mean.get(str(c))
        if v is None:
            missing.append(str(c))
        else:
            arr[i] = v
    return arr, missing


# ---------------------------------------------------------------------------
# CLI (server side; needs anndata)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, required=True)
    ap.add_argument("--splits_dir", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cell_col", default="cell_line")
    ap.add_argument("--pert_col", default="drug",
                    help="obs column holding the drug/perturbation label")
    ap.add_argument("--control_label", default="DMSO",
                    help="value in --pert_col marking control cells (e.g. DMSO or control)")
    ap.add_argument("--layer", default="counts",
                    help="adata layer with raw counts; falls back to .X if absent")
    ap.add_argument("--manifest_tpl", default="sciplex_{split}.parquet")
    ap.add_argument("--out_tpl", default="sciplex_{split}_control_counts.npy")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import anndata
    print(f"[rebuild] reading {args.h5ad}")
    adata = anndata.read_h5ad(args.h5ad)
    if args.layer and args.layer in adata.layers:
        X = adata.layers[args.layer]
    else:
        print(f"[rebuild] layer '{args.layer}' absent, using .X")
        X = adata.X
    counts = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    obs = adata.obs
    cell_lines = obs[args.cell_col].astype(str).to_numpy()
    is_control = (obs[args.pert_col].astype(str) == args.control_label).to_numpy()
    n_ctrl = int(is_control.sum())
    print(f"[rebuild] {counts.shape[0]} cells x {counts.shape[1]} genes; "
          f"{n_ctrl} control cells ('{args.pert_col}=={args.control_label}')")
    if n_ctrl == 0:
        raise ValueError(
            f"No control cells found with {args.pert_col}=={args.control_label!r}. "
            f"Check --pert_col / --control_label against adata.obs columns: {list(obs.columns)}")

    cl_means = cellline_control_means(counts, cell_lines, is_control)
    n_genes = counts.shape[1]
    print(f"[rebuild] per-cell-line DMSO pseudobulk for: {sorted(cl_means)}")

    for split in args.splits:
        man_path = args.splits_dir / args.manifest_tpl.format(split=split)
        out_path = args.splits_dir / args.out_tpl.format(split=split)
        man = pd.read_parquet(man_path)
        arr, missing = broadcast_control(cl_means, man[args.cell_col].astype(str), n_genes)
        if missing:
            raise ValueError(f"[{split}] no control mean for cell lines {sorted(set(missing))}")
        print(f"[rebuild] {split}: {arr.shape} (rows match manifest {len(man)}: {arr.shape[0] == len(man)})")
        if args.dry_run:
            continue
        if out_path.exists():
            bak = out_path.with_suffix(out_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(out_path, bak)
                print(f"[rebuild] backed up old -> {bak}")
        np.save(out_path, arr)
        print(f"[rebuild] wrote {out_path}")

    print("\n[rebuild] DONE. control_counts is now the cell-line DMSO pseudobulk mean. "
          "Re-run training/eval; Δ-vs-ridge is now apples-to-apples.")


if __name__ == "__main__":
    main()
