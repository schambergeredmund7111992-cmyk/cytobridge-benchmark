"""Compute pathway labels from the same frozen pair targets used for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_pair_gsea(
    true_logfc: np.ndarray,
    gene_ids: np.ndarray,
    gmt_path: Path,
    *,
    n_permutations: int = 1000,
) -> tuple[np.ndarray, list[str]]:
    """Return absolute normalized enrichment scores aligned to target rows."""
    values = np.asarray(true_logfc, dtype=float)
    genes = np.asarray(gene_ids).astype(str)
    if values.ndim != 2 or genes.ndim != 1 or values.shape[1] != len(genes):
        raise ValueError("true_logfc and gene_ids are not aligned.")
    if not np.isfinite(values).all() or len(set(genes.tolist())) != len(genes):
        raise ValueError("GSEA inputs must be finite with unique gene identifiers.")

    import gseapy as gp

    pathways = gp.parser.read_gmt(str(gmt_path))
    pathway_names = sorted(pathways)
    output = np.zeros((len(values), len(pathway_names)), dtype=np.float32)
    for row_index, row in enumerate(values):
        # Score ties are resolved by frozen gene order through stable mergesort.
        order = np.argsort(-row, kind="mergesort")
        ranked = pd.DataFrame({"gene": genes[order], "score": row[order]})
        result = gp.prerank(
            rnk=ranked,
            gene_sets=pathways,
            threads=1,
            permutation_num=n_permutations,
            seed=20260710 + row_index,
            no_plot=True,
            verbose=False,
        )
        nes = result.res2d.set_index("Term")["NES"].astype(float).to_dict()
        output[row_index] = [abs(float(nes.get(name, 0.0))) for name in pathway_names]
    denominator = np.maximum(output.max(axis=1, keepdims=True), 1.0e-8)
    return output / denominator, pathway_names


def expand_pair_values_to_manifest(
    pair_values: np.ndarray,
    pair_metadata: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    required = {"drug_id", "context_id"}
    if missing := required - set(pair_metadata.columns):
        raise ValueError(f"Pair metadata is missing columns: {sorted(missing)}")
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if len(pair_values) != len(pair_metadata):
        raise ValueError("Pair GSEA rows and pair metadata rows do not align.")
    keys = list(
        zip(
            pair_metadata["drug_id"].astype(str),
            pair_metadata["context_id"].astype(str),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("Pair metadata contains duplicate drug/context rows.")
    lookup = {key: pair_values[index] for index, key in enumerate(keys)}
    rows = []
    for drug, context in zip(manifest["drug_id"], manifest["context_id"]):
        key = (str(drug), str(context))
        if key not in lookup:
            raise ValueError(
                f"Manifest pair {key} is absent from frozen target metadata."
            )
        rows.append(lookup[key])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GSEA labels from frozen pair targets."
    )
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--gmt", type=Path, required=True)
    parser.add_argument("--prefix", choices=("sciplex", "tahoe"), default="sciplex")
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()
    split_dir = args.protocol_dir / "splits"
    pathway_names = None
    for split_name in ("train", "val", "test"):
        with np.load(
            split_dir / f"{split_name}_targets.npz", allow_pickle=False
        ) as data:
            true_logfc = data["true"]
            gene_ids = data["gene_ids"]
        pair_metadata = pd.read_csv(split_dir / f"{split_name}_targets_metadata.csv")
        manifest = pd.read_parquet(split_dir / f"{args.prefix}_{split_name}.parquet")
        pair_values, names = compute_pair_gsea(
            true_logfc,
            gene_ids,
            args.gmt,
            n_permutations=args.n_permutations,
        )
        if pathway_names is None:
            pathway_names = names
        elif names != pathway_names:
            raise RuntimeError("Pathway order changed between benchmark splits.")
        expanded = expand_pair_values_to_manifest(pair_values, pair_metadata, manifest)
        np.save(split_dir / f"{args.prefix}_{split_name}_pathway_gsea.npy", expanded)
    (args.protocol_dir / "pathway_names.txt").write_text(
        "\n".join(pathway_names or []) + "\n"
    )


if __name__ == "__main__":
    main()
