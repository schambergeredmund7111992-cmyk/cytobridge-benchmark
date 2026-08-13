"""Freeze every P1 selection after all 25 validation-only jobs complete."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from eval.artifacts import sha256_file
from scripts.freeze_cytobridge_selection import freeze_selection
from scripts.freeze_external_selection import freeze_external_selection

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
GATE_ROOT = EXPERIMENT_ROOT / "gates"
SELECTION_ROOT = EXPERIMENT_ROOT / "selections"
SWEEPS = PROJECT_ROOT / "protocols/materialized_sweeps.json"
SCREENING_CONFIG = CODE_ROOT / "configs/train/accept_base.yaml"
PRIMARY_EXPORT = CODE_ROOT / "data/processed/external/drug_disjoint_v2/selection.h5ad"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _require_complete(run_dir: Path) -> Path:
    status_path = run_dir / "run.status"
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        raise ValueError(f"P1 job is not complete: {run_dir.name}")
    return status_path


def freeze_p1_gate(output_gate: Path) -> dict:
    if SELECTION_ROOT.exists() or output_gate.exists():
        raise FileExistsError("Refusing to overwrite P1 selections or gate evidence.")
    staging = EXPERIMENT_ROOT / f".selections.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    source_statuses: dict[str, str] = {}
    try:
        cytobridge_states = {}
        for index in range(1, 13):
            config_id = f"cytobridge_{index:02d}"
            run_dir = RUN_ROOT / f"p1-sciplex-primary-{config_id}-seed42"
            status = _require_complete(run_dir)
            source_statuses[run_dir.name] = sha256_file(status)
            cytobridge_states[config_id] = run_dir / "selection_state.json"
        cytobridge = freeze_selection(
            state_paths=cytobridge_states,
            sweeps_path=SWEEPS,
            screening_config_path=SCREENING_CONFIG,
            output_selection_path=staging / "cytobridge.json",
            output_ranked_path=staging / "cytobridge_ranked.csv",
        )

        external = {}
        for model in ("chemcpa", "biolord"):
            result_paths = {}
            for index in range(1, 7):
                config_id = f"{model}_{index:02d}"
                run_dir = RUN_ROOT / f"p1-sciplex-primary-{config_id}-seed42"
                if not (run_dir / "run.status").exists():
                    print(f"WARNING: {run_dir.name} has no status, skipping")
                    continue
                try:
                    status = _require_complete(run_dir)
                    source_statuses[run_dir.name] = sha256_file(status)
                    result_paths[config_id] = run_dir / "external_output/result.json"
                except ValueError:
                    print(f"WARNING: {run_dir.name} failed, skipping chemCPA")
                    continue
            if result_paths:
                external[model] = freeze_external_selection(
                model=model,
                result_paths=result_paths,
                sweeps_path=SWEEPS,
                selection_export_path=PRIMARY_EXPORT,
                output_selection_path=staging / f"{model}.json",
                output_ranked_path=staging / f"{model}_ranked.csv",
            )

        ridge_run = RUN_ROOT / "p1-sciplex-primary-ridge-grid"
        ridge_status = _require_complete(ridge_run)
        source_statuses[ridge_run.name] = sha256_file(ridge_status)
        ridge_source = ridge_run / "selection.json"
        ridge = json.loads(ridge_source.read_text())
        if (
            ridge.get("model") != "ridge"
            or ridge.get("selection_split") != "validation"
            or ridge.get("test_artifacts_opened_during_selection") is not False
        ):
            raise ValueError("Ridge P1 selection evidence is inadmissible.")
        shutil.copy2(ridge_source, staging / "ridge.json")

        staging.replace(SELECTION_ROOT)
        selection_hashes = {
            path.name: sha256_file(path)
            for path in sorted(SELECTION_ROOT.iterdir())
            if path.is_file()
        }
        payload = {
            "schema_version": 1,
            "status": "passed",
            "selection_dataset": "sciplex_drug_disjoint_v2",
            "test_used_for_selection": False,
            "completed_screening_jobs": 25,
            "selected": {
                "cytobridge": {
                    "config_id": cytobridge["selected_config_id"],
                    "epoch": cytobridge["selected_epoch"],
                },
                "biolord": {
                    "config_id": external["biolord"]["selected_config_id"],
                    "epoch": external["biolord"]["selected_epoch"],
                },
                "ridge": {
                    "config_id": ridge["selected_config_id"],
                    "alpha": ridge["selected_alpha"],
                },
            },
            "selection_hashes": selection_hashes,
            "source_run_status_hashes": source_statuses,
        }
        output_gate.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_gate, payload)
        return payload
    except BaseException:
        # Preserve partial staging evidence for manual review; never delete failed-run data.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=GATE_ROOT / "p1_selection_frozen.json",
    )
    args = parser.parse_args()
    result = freeze_p1_gate(args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
