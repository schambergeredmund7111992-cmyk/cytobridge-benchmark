"""
eval/run_replogle.py
--------------------
Negative control on Replogle Perturb-seq.

Replogle is GENETIC perturbation (CRISPRi), not chemical. Our model is
trained on chemical perturbation. Therefore:
    - On Replogle, our model should perform NEAR-RANDOM.
    - If it performs well, that means we learned generic perturbation
      effects (e.g. cell stress) instead of drug-specific responses.
    - This is a critical sanity check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval.run_internal import aggregate_by_pair, load_model, predict_all
from eval.metrics import bootstrap_ci, per_pair_pearson, per_pair_spearman
from cytobridge.data import CytoBridgeDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/replogle/manifest.parquet"))
    parser.add_argument("--cell_emb", type=Path,
                        default=Path("data/cache/replogle_scgpt_emb.npy"))
    parser.add_argument("--drug_emb", type=Path,
                        default=Path("data/cache/replogle_molformer_emb.npz"),
                        help="Use a surrogate drug embedding (e.g. all zeros) "
                             "since Replogle has no real drugs.")
    parser.add_argument("--counts", type=Path,
                        default=Path("data/cache/replogle_treated_counts.npy"))
    parser.add_argument("--control_counts", type=Path,
                        default=Path("data/cache/replogle_control_counts.npy"))
    parser.add_argument("--gsea", type=Path,
                        default=Path("data/cache/replogle_pathway_gsea.npy"))
    parser.add_argument("--out", type=Path, default=Path("results/cytobridge_replogle.csv"))
    args = parser.parse_args()

    model = load_model(args.ckpt)
    ds = CytoBridgeDataset(
        manifest_path=args.manifest, cell_emb_path=args.cell_emb,
        drug_emb_path=args.drug_emb, treated_counts_path=args.counts,
        pathway_gsea_path=args.gsea,
        control_counts_path=args.control_counts if args.control_counts.exists() else None,
        n_hard_same_drug=0, n_hard_same_cell=0,
    )
    pred_mu, true_counts, ctrl_counts, drugs, cells = predict_all(model, ds)
    preds, trues, drugs, cells = aggregate_by_pair(
        pred_mu, true_counts, ctrl_counts, drugs, cells
    )
    pearson = per_pair_pearson(trues, preds)
    spearman = per_pair_spearman(trues, preds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"drug_id": drugs, "cell_line": cells,
                  "pearson_top50": pearson, "spearman_top50": spearman}).to_csv(args.out, index=False)

    s_mean, s_lo, s_hi = bootstrap_ci(spearman)
    print("\n=== Replogle Negative Control (CRISPRi, no chemicals) ===")
    print(f"Spearman@50: {s_mean:.4f}  [95% CI {s_lo:.4f}, {s_hi:.4f}]")
    if abs(s_mean) > 0.10:
        print("⚠️  WARNING: |Spearman| > 0.10 suggests model learned non-drug-specific "
              "perturbation. Add 'no-drug' hard negatives in training.")
    else:
        print("✓ Replogle Spearman near zero — drug-specificity confirmed.")


if __name__ == "__main__":
    main()
