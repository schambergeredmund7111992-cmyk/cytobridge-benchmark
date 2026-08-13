"""Aggregate validated scored artifacts across seeds and paired models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.artifacts import sha256_file, validate_artifact
from eval.statistics import (
    benjamini_hochberg,
    drug_cluster_bootstrap,
    paired_hierarchical_bootstrap,
    seed_drug_hierarchical_bootstrap,
)

SCORED_VALUE_COLUMNS = (
    "conditional_accuracy",
    "pair_own_spearman_top50",
)


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _validated_run_rows(manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = pd.read_csv(manifest_path)
    required = {"model", "seed", "artifact_dir", "scored_dir"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Run manifest is missing columns: {sorted(missing)}")
    if manifest[["model", "seed"]].astype(str).duplicated().any():
        raise ValueError("Run manifest model/seed identifiers must be unique.")
    base = manifest_path.parent
    pair_frames = []
    metric_rows = []
    input_hashes = {}
    for row in manifest.itertuples(index=False):
        model = str(row.model)
        training_seed = str(row.seed)
        artifact_dir = _resolve(base, row.artifact_dir)
        scored_dir = _resolve(base, row.scored_dir)
        artifact = validate_artifact(artifact_dir)
        metrics_path = scored_dir / "metrics.json"
        per_pair_path = scored_dir / "per_pair.csv"
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("artifact_predictions_sha256") != sha256_file(
            artifact_dir / "predictions.npz"
        ):
            raise ValueError(
                f"Scored predictions hash mismatch for {model}/{training_seed}."
            )
        if metrics.get("artifact_metadata_sha256") != sha256_file(
            artifact_dir / "metadata.csv"
        ):
            raise ValueError(
                f"Scored metadata hash mismatch for {model}/{training_seed}."
            )
        per_pair = pd.read_csv(per_pair_path)
        required_pair = {
            "pair_index",
            "pair_id",
            "drug_id",
            "context_id",
            *SCORED_VALUE_COLUMNS,
        }
        if missing_pair := required_pair - set(per_pair.columns):
            raise ValueError(
                f"Scored pair table for {model}/{training_seed} is missing "
                f"{sorted(missing_pair)}"
            )
        ordered = per_pair.sort_values("pair_index", kind="mergesort")
        if (
            ordered["pair_id"].astype(str).tolist()
            != artifact.metadata["pair_id"].tolist()
        ):
            raise ValueError(f"Scored pair order mismatch for {model}/{training_seed}.")
        per_pair = per_pair.copy()
        per_pair["model"] = model
        per_pair["seed"] = training_seed
        pair_frames.append(per_pair)
        metric_rows.append(
            {
                "model": model,
                "seed": training_seed,
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
        prefix = f"{model}:{training_seed}"
        input_hashes[f"{prefix}:artifact_predictions"] = sha256_file(
            artifact_dir / "predictions.npz"
        )
        input_hashes[f"{prefix}:artifact_metadata"] = sha256_file(
            artifact_dir / "metadata.csv"
        )
        input_hashes[f"{prefix}:metrics"] = sha256_file(metrics_path)
        input_hashes[f"{prefix}:per_pair"] = sha256_file(per_pair_path)
    return (
        pd.concat(pair_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
        input_hashes,
    )


def _estimate_model(
    frame: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int,
    bootstrap_seed: int,
):
    if frame["seed"].nunique() == 1:
        return drug_cluster_bootstrap(
            frame,
            value_col=value_col,
            n_boot=n_boot,
            seed=bootstrap_seed,
        )
    return seed_drug_hierarchical_bootstrap(
        frame,
        value_col=value_col,
        n_boot=n_boot,
        seed=bootstrap_seed,
    )


def aggregate_benchmark(
    manifest_path: Path,
    output_dir: Path,
    *,
    reference_model: str | None = None,
    n_boot: int = 10_000,
    bootstrap_seed: int = 7301,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate output: {output_dir}")
    per_pair, run_metrics, input_hashes = _validated_run_rows(manifest_path)
    leaderboard_rows = []
    draws_by_model = {}
    for model, frame in per_pair.groupby("model", sort=True):
        conditional = _estimate_model(
            frame,
            "conditional_accuracy",
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
        )
        spearman = _estimate_model(
            frame,
            "pair_own_spearman_top50",
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
        )
        draws_by_model[str(model)] = {
            "conditional": conditional.draws,
            "spearman": spearman.draws,
        }
        leaderboard_rows.append(
            {
                "model": str(model),
                "conditional_accuracy": conditional.estimate,
                "conditional_ci_low": conditional.ci_low,
                "conditional_ci_high": conditional.ci_high,
                "pair_own_spearman_top50": spearman.estimate,
                "spearman_ci_low": spearman.ci_low,
                "spearman_ci_high": spearman.ci_high,
                "n_seeds": int(frame["seed"].nunique()),
                "n_drugs": int(frame["drug_id"].nunique()),
                "n_contexts": int(frame["context_id"].nunique()),
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["conditional_accuracy", "pair_own_spearman_top50", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    comparisons = []
    skipped_unpaired_models = []
    if reference_model is not None:
        if reference_model not in set(per_pair["model"]):
            raise ValueError(
                f"Reference model {reference_model!r} is absent from the manifest."
            )
        reference_seeds = set(
            per_pair.loc[per_pair["model"].eq(reference_model), "seed"]
        )
        for model in sorted(set(per_pair["model"]) - {reference_model}):
            model_seeds = set(per_pair.loc[per_pair["model"].eq(model), "seed"])
            if model_seeds != reference_seeds:
                skipped_unpaired_models.append(
                    {
                        "model": model,
                        "reason": "seed_sets_differ",
                        "model_seeds": sorted(str(value) for value in model_seeds),
                        "reference_seeds": sorted(
                            str(value) for value in reference_seeds
                        ),
                    }
                )
                continue
            subset = per_pair[per_pair["model"].isin([model, reference_model])]
            record = {"model": model, "reference": reference_model}
            for value_col in SCORED_VALUE_COLUMNS:
                label = value_col
                effect = paired_hierarchical_bootstrap(
                    subset,
                    model,
                    reference_model,
                    value_col=value_col,
                    n_boot=n_boot,
                    seed=bootstrap_seed,
                )
                record[f"{label}_difference"] = effect.estimate
                record[f"{label}_difference_ci_low"] = effect.ci_low
                record[f"{label}_difference_ci_high"] = effect.ci_high
            comparisons.append(record)
    if "permutation_p_one_sided" in run_metrics:
        run_metrics["permutation_q_bh_across_runs"] = benjamini_hochberg(
            run_metrics["permutation_p_one_sided"].to_numpy(dtype=float)
        )

    output_dir.mkdir(parents=True)
    per_pair.to_csv(output_dir / "per_pair_all.csv", index=False)
    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    comparison_columns = [
        "model",
        "reference",
        *[
            f"{value_col}_{suffix}"
            for value_col in SCORED_VALUE_COLUMNS
            for suffix in (
                "difference",
                "difference_ci_low",
                "difference_ci_high",
            )
        ],
    ]
    pd.DataFrame(comparisons, columns=comparison_columns).to_csv(
        output_dir / "paired_vs_reference.csv", index=False
    )
    np.savez_compressed(
        output_dir / "leaderboard_bootstrap_draws.npz",
        **{
            f"{model}__{metric}": draws
            for model, metrics in sorted(draws_by_model.items())
            for metric, draws in sorted(metrics.items())
        },
    )
    summary = {
        "manifest_sha256": sha256_file(manifest_path),
        "input_hashes": input_hashes,
        "reference_model": reference_model,
        "n_bootstrap_draws": n_boot,
        "bootstrap_seed": bootstrap_seed,
        "n_models": int(leaderboard["model"].nunique()),
        "skipped_unpaired_models": skipped_unpaired_models,
        "validated_artifacts_only": True,
    }
    (output_dir / "aggregate_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-model")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    result = aggregate_benchmark(
        args.manifest,
        args.out,
        reference_model=args.reference_model,
        n_boot=args.n_boot,
        bootstrap_seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
