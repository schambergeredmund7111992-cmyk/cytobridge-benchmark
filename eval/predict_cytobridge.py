"""Generate row-aligned CytoBridge pair predictions for immutable packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cytobridge.data import CytoBridgeDataset
from eval.aggregation import aggregate_logfc_by_pair
from eval.package_artifact import load_targets
from eval.run_internal import load_model, predict_all


def align_to_frozen_targets(
    pair_pred: np.ndarray,
    reconstructed_true: np.ndarray,
    generated_metadata: pd.DataFrame,
    frozen_true: np.ndarray,
    frozen_metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if generated_metadata["pair_id"].astype(str).duplicated().any():
        raise ValueError("Generated CytoBridge pair identifiers are not unique.")
    generated = generated_metadata.set_index("pair_id")
    pair_ids = frozen_metadata["pair_id"].astype(str).to_numpy()
    if set(generated.index.astype(str)) != set(pair_ids):
        raise ValueError("CytoBridge generated pairs differ from frozen targets.")
    order = generated.loc[pair_ids, "pair_index"].to_numpy(dtype=int)
    pred = np.asarray(pair_pred)[order]
    observed_true = np.asarray(reconstructed_true)[order]
    if observed_true.shape != frozen_true.shape or not np.allclose(
        observed_true, frozen_true, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            "CytoBridge true-count reconstruction differs from frozen targets."
        )
    return pred, pair_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-emb", type=Path, required=True)
    parser.add_argument("--drug-emb", type=Path, required=True)
    parser.add_argument("--treated-counts", type=Path, required=True)
    parser.add_argument("--input-control-counts", type=Path, required=True)
    parser.add_argument("--truth-control-counts", type=Path, required=True)
    parser.add_argument("--pathway-gsea", type=Path, required=True)
    parser.add_argument("--frozen-targets", type=Path, required=True)
    parser.add_argument("--frozen-metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(
            f"Refusing to overwrite CytoBridge predictions: {args.out}"
        )
    model = load_model(args.ckpt)
    dataset = CytoBridgeDataset(
        manifest_path=args.manifest,
        cell_emb_path=args.cell_emb,
        drug_emb_path=args.drug_emb,
        treated_counts_path=args.treated_counts,
        pathway_gsea_path=args.pathway_gsea,
        input_control_counts_path=args.input_control_counts,
        truth_control_counts_path=args.truth_control_counts,
        n_hard_same_drug=0,
        n_hard_same_cell=0,
    )
    pred_mu, treated, truth_control, drugs, contexts = predict_all(
        model,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if not np.isfinite(pred_mu).all() or (pred_mu < 0).any():
        raise FloatingPointError("CytoBridge produced invalid non-count predictions.")
    pair_pred, reconstructed_true, generated_metadata = aggregate_logfc_by_pair(
        pred_mu,
        treated,
        truth_control,
        drugs,
        contexts,
    )
    frozen_true, gene_ids = load_targets(args.frozen_targets)
    frozen_metadata = pd.read_csv(args.frozen_metadata)
    pred, pair_ids = align_to_frozen_targets(
        pair_pred,
        reconstructed_true,
        generated_metadata,
        frozen_true,
        frozen_metadata,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pred=pred.astype(np.float32),
        pair_ids=np.asarray(pair_ids, dtype=str),
        gene_ids=gene_ids,
    )
    print(
        json.dumps(
            {
                "pairs": len(pred),
                "genes": len(gene_ids),
                "frozen_truth_reconstruction_verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
