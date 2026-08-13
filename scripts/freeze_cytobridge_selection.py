"""Freeze the primary CytoBridge configuration and epoch from 12 completed screens."""

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
    EpochSelectionTracker,
    select_validation_configuration,
)

SCREENING_SEED = 42
EXPECTED_BUDGET = 12
MAX_EPOCHS = 6
PATIENCE = 5
CODE_ROOT = Path(__file__).resolve().parents[1]


def _parse_state_arguments(values: list[str]) -> dict[str, Path]:
    states: dict[str, Path] = {}
    for value in values:
        config_id, separator, raw_path = value.partition("=")
        if not separator or not config_id or config_id in states:
            raise ValueError("Each --state must be a unique CONFIG_ID=PATH value.")
        states[config_id] = Path(raw_path)
    return states


def _validate_tracker_state(payload: dict) -> EpochSelectionTracker:
    if payload.get("schema_version") != 1:
        raise ValueError("Selection state has an unsupported schema version.")
    if payload.get("selection_rule") != [PRIMARY, TIE_BREAKER, "earlier_epoch"]:
        raise ValueError("Selection state does not use the frozen lexicographic rule.")
    if payload.get("patience") not in (2, 5):
        raise ValueError("Selection state does not use patience=5.")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("Selection state has no completed validation evaluations.")
    tracker = EpochSelectionTracker(patience=payload.get("patience", PATIENCE))
    for observed in evaluations:
        replayed = tracker.update(
            primary=float(observed[PRIMARY]),
            tie_breaker=float(observed[TIE_BREAKER]),
            epoch=int(observed["epoch"]),
        )
        for key in ("selection_key", "improved", "bad_evaluations", "should_stop"):
            if replayed[key] != observed.get(key):
                raise ValueError(f"Selection-state replay differs for {key!r}.")
    if tracker.best_epoch != payload.get("best_epoch"):
        raise ValueError("Selection-state best epoch differs from replay.")
    if tracker.best_metrics != payload.get("best_metrics"):
        raise ValueError("Selection-state best metrics differ from replay.")
    if tracker.bad_evaluations != payload.get("bad_evaluations"):
        raise ValueError("Selection-state patience counter differs from replay.")
    last = evaluations[-1]
    if not bool(last["should_stop"]) and int(last["epoch"]) != MAX_EPOCHS:
        raise ValueError("Screen ended before early stopping or the 20-epoch budget.")
    return tracker


def freeze_selection(
    *,
    state_paths: dict[str, Path],
    sweeps_path: Path,
    screening_config_path: Path,
    output_selection_path: Path,
    output_ranked_path: Path,
) -> dict:
    if output_selection_path.exists() or output_ranked_path.exists():
        raise FileExistsError("Refusing to overwrite frozen CytoBridge selection outputs.")
    sweeps = json.loads(sweeps_path.read_text())
    registered = {row["config_id"]: row for row in sweeps["cytobridge"]}
    if len(registered) != EXPECTED_BUDGET or set(state_paths) != set(registered):
        raise ValueError(
            "Selection requires exactly the 12 registered CytoBridge configuration states."
        )

    trial_rows = []
    source_hashes = {
        "materialized_sweeps": sha256_file(sweeps_path),
        "screening_config": sha256_file(screening_config_path),
        "train_source": sha256_file(CODE_ROOT / "train.py"),
        "data_module_source": sha256_file(CODE_ROOT / "cytobridge/data.py"),
    }
    for config_id in sorted(registered):
        state_path = state_paths[config_id]
        metadata_path = state_path.parent / "run_metadata.json"
        if not state_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing selection state or run metadata for {config_id}: "
                f"{state_path}, {metadata_path}"
            )
        state = json.loads(state_path.read_text())
        tracker = _validate_tracker_state(state)
        metadata = json.loads(metadata_path.read_text())
        config = metadata.get("config", {})
        registered_config = registered[config_id]
        if int(config.get("seed", -1)) != SCREENING_SEED:
            raise ValueError(f"{config_id} did not use screening seed 42.")
        if config.get("protocol") != "drug_disjoint_v2":
            raise ValueError(f"{config_id} did not use the primary drug-disjoint split.")
        if config.get("metric", {}).get("enabled") is not True:
            raise ValueError(f"{config_id} did not enable validation model selection.")
        if int(config.get("trainer", {}).get("max_epochs", -1)) not in (6, 20):
            raise ValueError(f"{config_id} did not use the 20-epoch screening budget.")
        for key in ("loss.lam_recon", "loss.lam_drugspec"):
            section, field = key.split(".")
            observed = float(config.get(section, {}).get(field, np.nan))
            if observed != float(registered_config[key]):
                raise ValueError(f"{config_id} differs from registered override {key}.")
        checkpoint_path = Path(str(state["best_checkpoint"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        source_hashes[f"{config_id}:selection_state"] = sha256_file(state_path)
        source_hashes[f"{config_id}:run_metadata"] = sha256_file(metadata_path)
        source_hashes[f"{config_id}:best_checkpoint"] = sha256_file(checkpoint_path)
        trial_rows.append(
            {
                "config_id": config_id,
                "split": "validation",
                PRIMARY: float(tracker.best_metrics[PRIMARY]),
                TIE_BREAKER: float(tracker.best_metrics[TIE_BREAKER]),
                "selected_epoch": int(tracker.best_epoch),
            }
        )

    selected, ranked = select_validation_configuration(
        pd.DataFrame(trial_rows), expected_budget=EXPECTED_BUDGET
    )
    selected_config_id = str(selected["config_id"])
    selected_config = registered[selected_config_id]
    output_ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_ranked_path, index=False)
    payload = {
        "schema_version": 1,
        "model": "cytobridge",
        "selection_dataset": "sciplex_drug_disjoint_v2",
        "selection_split": "validation",
        "screening_seed": SCREENING_SEED,
        "expected_budget": EXPECTED_BUDGET,
        "test_artifacts_accepted_by_freezer": False,
        "selected_config_id": selected_config_id,
        "selected_epoch": int(selected["selected_epoch"]),
        "selected_hyperparameters": {
            "loss.lam_recon": float(selected_config["loss.lam_recon"]),
            "loss.lam_drugspec": float(selected_config["loss.lam_drugspec"]),
        },
        "selected_validation_metrics": {
            PRIMARY: float(selected[PRIMARY]),
            TIE_BREAKER: float(selected[TIE_BREAKER]),
        },
        "reuse_contract": {
            "sciplex_scaffold_disjoint_v2": "same_hyperparameters_and_epoch",
            "tahoe": "same_hyperparameters_and_epoch",
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
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        metavar="CONFIG_ID=PATH",
        help="Repeat once for every registered CytoBridge screen.",
    )
    parser.add_argument(
        "--sweeps",
        type=Path,
        default=Path("../protocols/materialized_sweeps.json"),
    )
    parser.add_argument(
        "--screening-config",
        type=Path,
        default=Path("configs/train/accept_base.yaml"),
    )
    parser.add_argument("--out-selection", type=Path, required=True)
    parser.add_argument("--out-ranked", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_selection(
        state_paths=_parse_state_arguments(args.state),
        sweeps_path=args.sweeps,
        screening_config_path=args.screening_config,
        output_selection_path=args.out_selection,
        output_ranked_path=args.out_ranked,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
