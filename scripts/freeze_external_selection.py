"""Freeze chemCPA or biolord configuration/epoch from validation-only screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.artifacts import sha256_file
from eval.model_selection import (
    PRIMARY,
    TIE_BREAKER,
    epoch_selection_key,
    select_validation_configuration,
)

MODEL_SPECS = {
    "chemcpa": {
        "config_prefix": "chemcpa_",
        "expected_epochs": [50, 100, 150, 200, 201],
        "max_epochs": 201,
        "patience": None,
    },
    "biolord": {
        "config_prefix": "biolord_",
        "expected_epochs": None,
        "max_epochs": 500,
        "patience": 20,
    },
}


def _parse_result_arguments(values: list[str]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for value in values:
        config_id, separator, raw_path = value.partition("=")
        if not separator or not config_id or config_id in results:
            raise ValueError("Each --result must be a unique CONFIG_ID=PATH value.")
        results[config_id] = Path(raw_path)
    return results


def _validate_evaluations(model: str, result: dict) -> tuple[int, dict]:
    spec = MODEL_SPECS[model]
    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError(f"{model} result contains no validation evaluations.")
    epochs = [int(row["epoch"]) for row in evaluations]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"{model} validation epochs are not strictly increasing.")
    if spec["expected_epochs"] is not None and epochs != spec["expected_epochs"]:
        raise ValueError(f"{model} did not complete every registered validation epoch.")
    if model == "biolord" and any(epoch % 20 or epoch > 500 for epoch in epochs):
        raise ValueError("biolord evaluated outside the registered 20-epoch cadence.")
    keys = []
    for row in evaluations:
        values = np.asarray([row[PRIMARY], row[TIE_BREAKER]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{model} validation metrics contain NaN or Inf.")
        keys.append(epoch_selection_key(values[0], values[1], int(row["epoch"])))
    best_index = max(range(len(keys)), key=keys.__getitem__)
    best_row = evaluations[best_index]
    best = result.get("best")
    if (
        not isinstance(best, dict)
        or int(best.get("epoch", -1)) != int(best_row["epoch"])
        or best.get("selection_key") != list(keys[best_index])
    ):
        raise ValueError(f"{model} recorded best state differs from replay.")
    for metric in (PRIMARY, TIE_BREAKER):
        if float(best.get("metrics", {}).get(metric, np.nan)) != float(best_row[metric]):
            raise ValueError(f"{model} recorded best metrics differ from replay.")
    if int(result.get("selected_epoch", -1)) != int(best_row["epoch"]):
        raise ValueError(f"{model} selected epoch differs from replay.")
    if model == "biolord" and epochs[-1] < int(spec["max_epochs"]):
        patience = int(spec["patience"])
        if len(evaluations) - best_index - 1 < patience:
            raise ValueError("biolord stopped before 20 non-improving evaluations.")
    return int(best_row["epoch"]), {
        PRIMARY: float(best_row[PRIMARY]),
        TIE_BREAKER: float(best_row[TIE_BREAKER]),
    }


def freeze_external_selection(
    *,
    model: str,
    result_paths: dict[str, Path],
    sweeps_path: Path,
    selection_export_path: Path,
    output_selection_path: Path,
    output_ranked_path: Path,
) -> dict:
    if model not in MODEL_SPECS:
        raise ValueError(f"Unsupported external model: {model}")
    if output_selection_path.exists() or output_ranked_path.exists():
        raise FileExistsError("Refusing to overwrite frozen external selection outputs.")
    sweeps = json.loads(sweeps_path.read_text())
    registered = {row["config_id"]: row for row in sweeps[model]}
    if len(registered) != 6 or set(result_paths) != set(registered):
        raise ValueError(f"{model} selection requires exactly its six registered trials.")
    export_manifest_path = selection_export_path.with_suffix(".manifest.json")
    export_manifest = json.loads(export_manifest_path.read_text())
    if (
        export_manifest.get("included_benchmark_splits") != ["train", "validation"]
        or export_manifest.get("test_responses_included") is not False
        or export_manifest.get("train_validation_refit_union") is not False
    ):
        raise ValueError("External screening export does not physically exclude test rows.")
    export_hash = sha256_file(selection_export_path)
    sweep_hash = sha256_file(sweeps_path)
    trial_rows = []
    source_hashes = {
        "materialized_sweeps": sweep_hash,
        "selection_export": export_hash,
        "selection_export_manifest": sha256_file(export_manifest_path),
    }
    for config_id in sorted(registered):
        result_path = result_paths[config_id]
        result = json.loads(result_path.read_text())
        if (
            result.get("schema_version") != 1
            or result.get("config_id") != config_id
            or int(result.get("seed", -1)) != 42
            or result.get("mode") != "selection"
        ):
            raise ValueError(f"{config_id} result metadata differs from screening protocol.")
        if result.get("source_hashes", {}).get("export") != export_hash:
            raise ValueError(f"{config_id} did not use the frozen selection export.")
        if result.get("source_hashes", {}).get("sweeps") != sweep_hash:
            raise ValueError(f"{config_id} did not use the frozen sweep file.")
        if model == "chemcpa" and (
            result.get("initialization") != "from_scratch"
            or result.get("pretrained_weights_used") is not False
        ):
            raise ValueError("chemCPA selection used inadmissible initialization.")
        selected_epoch, metrics = _validate_evaluations(model, result)
        trial_rows.append(
            {
                "config_id": config_id,
                "split": "validation",
                PRIMARY: metrics[PRIMARY],
                TIE_BREAKER: metrics[TIE_BREAKER],
                "selected_epoch": selected_epoch,
            }
        )
        source_hashes[f"{config_id}:result"] = sha256_file(result_path)
    selected, ranked = select_validation_configuration(
        pd.DataFrame(trial_rows), expected_budget=6
    )
    selected_config_id = str(selected["config_id"])
    output_ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_ranked_path, index=False)
    payload = {
        "schema_version": 1,
        "model": model,
        "selection_dataset": "sciplex_drug_disjoint_v2",
        "selection_split": "validation",
        "screening_seed": 42,
        "expected_budget": 6,
        "test_rows_present_in_selection_export": False,
        "selected_config_id": selected_config_id,
        "selected_epoch": int(selected["selected_epoch"]),
        "selected_hyperparameters": registered[selected_config_id],
        "selected_validation_metrics": {
            PRIMARY: float(selected[PRIMARY]),
            TIE_BREAKER: float(selected[TIE_BREAKER]),
        },
        "reuse_contract": {
            "sciplex_scaffold_disjoint_v2": "same_hyperparameters_and_epoch",
            "retuning_allowed": False,
        },
        "ranked_trials_sha256": sha256_file(output_ranked_path),
        "source_hashes": source_hashes,
    }
    output_selection_path.parent.mkdir(parents=True, exist_ok=True)
    output_selection_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="CONFIG_ID=PATH",
    )
    parser.add_argument(
        "--sweeps",
        type=Path,
        default=Path("../protocols/materialized_sweeps.json"),
    )
    parser.add_argument("--selection-export", type=Path, required=True)
    parser.add_argument("--out-selection", type=Path, required=True)
    parser.add_argument("--out-ranked", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_external_selection(
        model=args.model,
        result_paths=_parse_result_arguments(args.result),
        sweeps_path=args.sweeps,
        selection_export_path=args.selection_export,
        output_selection_path=args.out_selection,
        output_ranked_path=args.out_ranked,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
