#!/usr/bin/env python
"""Create a tiny deterministic CytoBridge dataset for local smoke checks.

The generated files follow the same contract as the real sci-Plex pipeline,
but use CSV manifests and small NumPy arrays so students can validate the
model/data path before downloading large public datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CELL_LINES = ["A549", "MCF7", "K562", "HT29"]
SPLITS = {
    "train": ["drug_a", "drug_b"],
    "val": ["drug_c"],
    "test": ["drug_d"],
}


def _make_manifest(split: str, drugs: list[str]) -> pd.DataFrame:
    rows = []
    offset = {"train": 100, "val": 200, "test": 300}[split]
    for drug in drugs:
        for cell_idx, cell_line in enumerate(CELL_LINES):
            rows.append(
                {
                    "cell_idx": offset + len(rows),
                    "control_cell_idx": cell_idx,
                    "drug_id": drug,
                    "cell_line": cell_line,
                }
            )
    return pd.DataFrame(rows)


def create_smoke_data(out_dir: Path, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    n_cell_tokens = 5
    d_cell = 8
    n_drug_tokens = 3
    d_drug = 6
    n_genes = 10
    n_pathways = 5

    cell_emb = rng.normal(size=(len(CELL_LINES), n_cell_tokens, d_cell)).astype("float32")
    np.save(out_dir / "sciplex_scgpt_emb.npy", cell_emb)

    drug_ids = [drug for drugs in SPLITS.values() for drug in drugs]
    drug_tokens = rng.normal(size=(len(drug_ids), n_drug_tokens, d_drug)).astype("float32")
    drug_masks = np.ones((len(drug_ids), n_drug_tokens), dtype=bool)
    np.savez(
        out_dir / "sciplex_molformer_emb.npz",
        tokens=drug_tokens,
        masks=drug_masks,
        drug_ids=np.asarray(drug_ids),
    )

    split_meta: dict[str, list[str]] = {}
    for split, drugs in SPLITS.items():
        manifest = _make_manifest(split, drugs)
        manifest.to_csv(splits_dir / f"sciplex_{split}.csv", index=False)
        split_meta[f"{split}_drugs"] = drugs

        n_rows = len(manifest)
        base_counts = rng.poisson(lam=4, size=(n_rows, n_genes)).astype("float32")
        drug_effect = np.linspace(0.2, 1.1, n_genes, dtype="float32")
        treated = base_counts + drug_effect
        control = base_counts
        pathway = rng.random(size=(n_rows, n_pathways)).astype("float32")

        np.save(splits_dir / f"sciplex_{split}_treated_counts.npy", treated)
        np.save(splits_dir / f"sciplex_{split}_control_counts.npy", control)
        np.save(splits_dir / f"sciplex_{split}_pathway_gsea.npy", pathway)

    (splits_dir / "internal_splits.json").write_text(json.dumps(split_meta, indent=2))
    pd.DataFrame({"gene": [f"gene_{i}" for i in range(n_genes)]}).to_csv(
        out_dir / "genes.csv", index=False
    )
    pd.DataFrame({"pathway": [f"pathway_{i}" for i in range(n_pathways)]}).to_csv(
        out_dir / "pathways.csv", index=False
    )
    print(f"[smoke-data] wrote tiny dataset to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    create_smoke_data(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
