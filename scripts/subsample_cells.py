"""Subsample <=cap cells per (drug_id, cell_line) pair, per split, to make
training tractable on the larger independent split (442k cells ->
~tens of k). Writes a parallel splits dir; the scGPT cache is UNTOUCHED (it is
looked up by control_cell_idx, which the subsampled manifest preserves).

Correctness (mirrors cytobridge/data.py indexing):
  - treated/control counts and pathway_gsea are POSITIONALLY aligned with the
    manifest rows -> subset by the same row positions.
  - cell_emb is indexed by manifest['control_cell_idx'] -> cache stays full.
  - internal_splits.json is drug-level -> copied unchanged (splits stay disjoint).
Counts are cast to float32 (raw single-cell counts are exact in float32; halves
I/O vs float64), which is what dominated the dataloader stall.

  python scripts/subsample_cells.py \
    --splits_dir data/processed/sciplex/splits \
    --out_dir   data/processed/sciplex/splits_sub --cap 150
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def select_capped_indices(man: pd.DataFrame, cap: int, seed: int = 42) -> np.ndarray:
    """Row positions to keep: <=cap rows per (drug_id, cell_line), deterministic.
    man must have a 0..N-1 RangeIndex (call .reset_index(drop=True) first)."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, grp in man.groupby(["drug_id", "cell_line"], sort=True):
        idx = grp.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())
    return np.array(sorted(keep), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sp = Path(args.splits_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        man = pd.read_parquet(sp / f"sciplex_{split}.parquet").reset_index(drop=True)
        keep = select_capped_indices(man, args.cap, seed=args.seed)

        man_sub = man.iloc[keep].reset_index(drop=True)
        man_sub.to_parquet(out / f"sciplex_{split}.parquet")
        for kind in ["treated", "control"]:
            arr = np.load(sp / f"sciplex_{split}_{kind}_counts.npy", mmap_mode="r")
            np.save(out / f"sciplex_{split}_{kind}_counts.npy",
                    np.asarray(arr[keep], dtype=np.float32))
        pg = np.load(sp / f"sciplex_{split}_pathway_gsea.npy", mmap_mode="r")
        np.save(out / f"sciplex_{split}_pathway_gsea.npy",
                np.asarray(pg[keep], dtype=np.float32))
        n_pairs = man.groupby(["drug_id", "cell_line"]).ngroups
        print(f"{split}: {len(man)} -> {len(keep)} rows over {n_pairs} pairs (cap {args.cap})")

    shutil.copy(sp / "internal_splits.json", out / "internal_splits.json")
    print("subsample done ->", out)


if __name__ == "__main__":
    main()
