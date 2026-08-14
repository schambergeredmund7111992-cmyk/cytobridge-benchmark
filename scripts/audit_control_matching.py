#!/usr/bin/env python
"""
audit_control_matching.py
-------------------------
Stage-0.5 cheap confound check (Codex TIER 2.3): the current pipeline matches
each treated cell to a RANDOM same-cell-line DMSO control
(data/preprocess.py:127-143, make_manifest -> rng.choice(candidates)).
It does NOT match by plate/well/replicate, so plate/batch effects could be a
parallel reason v1 underperforms ridge.

This script does NOT modify anything. It answers one question:
  "Is finer-grained (same-plate / same-well / same-replicate) control matching
   even POSSIBLE with the fields available in adata.obs?"

If yes  -> arm B (rebuild control with matched batch) is worth one run.
If no   -> drop arm B; the control-matching confound is not addressable and
           we focus compute on the delta-objective fix (arm C).

Run on the remote where the processed h5ad lives:

    cd code
    python scripts/audit_control_matching.py \
        --h5ad data/processed/sciplex_accept/drug_disjoint_v2/sciplex_processed.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# obs column name fragments that would let us match controls at a finer grain
BATCH_HINTS = ["plate", "well", "replicate", "rep", "batch", "lane",
               "hash", "channel", "sample", "donor", "experiment"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, type=Path)
    ap.add_argument("--drug_col", default="drug")
    ap.add_argument("--cell_col", default="cell_line")
    ap.add_argument("--control_label", default="DMSO")
    args = ap.parse_args()

    import anndata as ad
    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    print(f"[audit] {args.h5ad.name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"[audit] obs columns ({len(obs.columns)}):")
    for c in obs.columns:
        nun = obs[c].nunique(dropna=True)
        print(f"    - {c:30s} nunique={nun}")

    print("\n[audit] candidate batch/plate/well fields for finer control matching:")
    found = []
    for c in obs.columns:
        cl = c.lower()
        if any(h in cl for h in BATCH_HINTS):
            nun = obs[c].nunique(dropna=True)
            found.append((c, nun))
            print(f"    >>> {c:30s} nunique={nun}  example={list(pd.unique(obs[c].dropna()))[:4]}")
    if not found:
        print("    (none found)")

    # How many DMSO controls per (cell_line) and per (cell_line x candidate-batch)?
    print(f"\n[audit] DMSO control availability (control_label={args.control_label}):")
    is_ctrl = obs[args.drug_col].astype(str) == args.control_label
    print(f"    total control cells: {int(is_ctrl.sum())}")
    print(f"    controls per {args.cell_col}:")
    print(obs[is_ctrl].groupby(args.cell_col).size().to_string())

    for c, _ in found:
        # can we match a treated cell to a control sharing BOTH cell_line and this field?
        keys = [args.cell_col, c]
        ctrl_groups = obs[is_ctrl].groupby(keys).size()
        treated_groups = obs[~is_ctrl].groupby(keys).size()
        covered = treated_groups.index.isin(ctrl_groups.index)
        frac = float(np.mean(covered)) if len(covered) else 0.0
        print(f"\n[audit] if we match on ({args.cell_col}, {c}): "
              f"{frac*100:.1f}% of treated groups have >=1 control in the same group "
              f"({covered.sum()}/{len(covered)} groups)")

    print("\n[audit] DECISION RULE:")
    print("  - If a plate/well/replicate field exists AND >~80% of treated groups have a")
    print("    same-(cell_line, field) control -> arm B worth ONE run (rebuild matched control).")
    print("  - Otherwise drop arm B; spend compute on arm C (delta-objective v2).")


if __name__ == "__main__":
    main()
