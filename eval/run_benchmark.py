"""Score one validated prediction artifact under the frozen benchmark protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.artifacts import sha256_file, sha256_json, validate_artifact
from eval.metrics import _finite_correlation, conditional_rank_score, pair_own_spearman
from eval.statistics import (
    conditional_rank_permutation_test,
    drug_cluster_bootstrap,
)


def score_artifact(
    artifact_dir: Path,
    gene_panels_path: Path,
    output_dir: Path,
    *,
    n_boot: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 7301,
) -> dict:
    bundle = validate_artifact(artifact_dir)
    panel_payload = json.loads(gene_panels_path.read_text())
    normalized_panels = {
        str(context): [int(index) for index in indices]
        for context, indices in sorted(panel_payload.items())
    }
    observed_panel_hash = sha256_json(normalized_panels)
    expected_panel_hash = bundle.provenance["response_panel_hash"]
    if observed_panel_hash != expected_panel_hash:
        raise ValueError(
            "Frozen response panel hash does not match artifact provenance: "
            f"{observed_panel_hash} != {expected_panel_hash}."
        )
    conditional = conditional_rank_score(
        bundle.pred,
        bundle.true,
        bundle.metadata["context_id"],
        bundle.metadata["drug_id"],
        normalized_panels,
    )
    spearman = pair_own_spearman(bundle.true, bundle.pred, top_k=50)
    all_gene_pearson = np.asarray(
        [
            _finite_correlation(prediction, truth, "pearson")
            for prediction, truth in zip(bundle.pred, bundle.true)
        ],
        dtype=float,
    )
    artifact_metadata = bundle.metadata.reset_index(names="artifact_row")
    supplemental_columns = [
        column
        for column in artifact_metadata.columns
        if column not in {"drug_id", "context_id"}
    ]
    per_pair = conditional.per_pair.merge(
        artifact_metadata[supplemental_columns],
        left_on="pair_index",
        right_on="artifact_row",
        validate="one_to_one",
    )
    per_pair["pair_own_spearman_top50"] = spearman[per_pair["artifact_row"].to_numpy()]
    per_pair["all_gene_pearson"] = all_gene_pearson[per_pair["artifact_row"].to_numpy()]
    conditional_bootstrap = drug_cluster_bootstrap(
        per_pair,
        value_col="conditional_accuracy",
        n_boot=n_boot,
        seed=seed,
    )
    spearman_bootstrap = drug_cluster_bootstrap(
        per_pair,
        value_col="pair_own_spearman_top50",
        n_boot=n_boot,
        seed=seed,
    )
    permutation = conditional_rank_permutation_test(
        bundle.pred,
        bundle.true,
        bundle.metadata["context_id"],
        bundle.metadata["drug_id"],
        normalized_panels,
        n_permutations=n_permutations,
        seed=seed,
    )
    per_drug = per_pair.groupby("drug_id", sort=True, as_index=False).agg(
        conditional_accuracy=("conditional_accuracy", "mean"),
        transposed_rank=("transposed_rank", "mean"),
        similarity_gap=("similarity_gap", "mean"),
        pair_own_spearman_top50=("pair_own_spearman_top50", "mean"),
        all_gene_pearson=("all_gene_pearson", "mean"),
        n_contexts=("context_id", "nunique"),
    )
    metrics = {
        **conditional.summary,
        "pair_own_spearman_top50_drug_macro": float(
            per_drug["pair_own_spearman_top50"].mean()
        ),
        "all_gene_pearson_drug_macro": float(per_drug["all_gene_pearson"].mean()),
        "conditional_accuracy_ci95": [
            conditional_bootstrap.ci_low,
            conditional_bootstrap.ci_high,
        ],
        "pair_own_spearman_top50_ci95": [
            spearman_bootstrap.ci_low,
            spearman_bootstrap.ci_high,
        ],
        "permutation_p_one_sided": permutation.p_value,
        "permutation_exhaustive": permutation.exhaustive,
        "n_permutation_draws": permutation.n_permutations,
        "bootstrap_unit": "drug",
        "n_bootstrap_draws": n_boot,
        "seed": seed,
        "artifact_predictions_sha256": sha256_file(artifact_dir / "predictions.npz"),
        "artifact_metadata_sha256": sha256_file(artifact_dir / "metadata.csv"),
        "response_panel_sha256": observed_panel_hash,
    }
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark output: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    per_pair.to_csv(output_dir / "per_pair.csv", index=False)
    per_drug.to_csv(output_dir / "per_drug.csv", index=False)
    np.save(output_dir / "bootstrap_conditional.npy", conditional_bootstrap.draws)
    np.save(output_dir / "bootstrap_spearman.npy", spearman_bootstrap.draws)
    np.save(output_dir / "permutation_null.npy", permutation.draws)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a validated prediction artifact."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gene-panels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    metrics = score_artifact(
        args.artifact,
        args.gene_panels,
        args.out,
        n_boot=args.n_boot,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
