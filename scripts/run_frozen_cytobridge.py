"""Run a fixed-epoch CytoBridge refit from the frozen primary selection evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval.artifacts import sha256_file

CODE_ROOT = Path(__file__).resolve().parents[1]
FINAL_SEEDS = {11, 23, 42, 67, 101}
DATASETS = {
    "sciplex_drug_disjoint_v2": ("accept_final_refit", "drug_disjoint_v2"),
    "sciplex_scaffold_disjoint_v2": ("accept_final_refit", "scaffold_disjoint_v2"),
    "tahoe": ("accept_tahoe_final_refit", "tahoe"),
}
ARMS = {
    "cytobridge_full": {},
    "cytobridge_reconstruction_free": {
        "loss.lam_recon": 0.0,
        "loss.lam_logfc": 0.5,
        "loss.lam_norm_recon": 0.0,
    },
    "cytobridge_normalized_reconstruction": {
        "loss.lam_recon": 0.0,
        "loss.lam_logfc": 0.0,
        "loss.lam_norm_recon": 1.0,
    },
    "cytobridge_no_pathway_gate": {
        "model.use_pathway_gate": False,
        "loss.lam_pathway": 0.0,
        "loss.lam_kl": 0.0,
    },
    "cytobridge_drug_blind": {"data.randomize_drug_emb": True},
}


def _hydra_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def load_selection(selection_path: Path, sweeps_path: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    sweeps = json.loads(sweeps_path.read_text())
    registered = {row["config_id"]: row for row in sweeps["cytobridge"]}
    config_id = selection.get("selected_config_id")
    if (
        selection.get("schema_version") != 1
        or selection.get("model") != "cytobridge"
        or selection.get("selection_dataset") != "sciplex_drug_disjoint_v2"
        or selection.get("selection_split") != "validation"
        or selection.get("expected_budget") != 12
        or selection.get("test_artifacts_accepted_by_freezer") is not False
        or config_id not in registered
    ):
        raise ValueError("Primary selection evidence is not admissible.")
    expected_sweep_hash = selection.get("source_hashes", {}).get("materialized_sweeps")
    if expected_sweep_hash != sha256_file(sweeps_path):
        raise ValueError("Materialized sweep hash differs from frozen selection evidence.")
    registered_config = registered[str(config_id)]
    expected_hyperparameters = {
        "loss.lam_recon": float(registered_config["loss.lam_recon"]),
        "loss.lam_drugspec": float(registered_config["loss.lam_drugspec"]),
    }
    if selection.get("selected_hyperparameters") != expected_hyperparameters:
        raise ValueError("Frozen hyperparameters differ from the registered configuration.")
    epoch = selection.get("selected_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= 20:
        raise ValueError("Frozen epoch must be an integer in the screening budget [1, 20].")
    reuse = selection.get("reuse_contract", {})
    if reuse.get("retuning_allowed") is not False:
        raise ValueError("Frozen selection evidence does not prohibit retuning.")
    return selection


def load_runtime_decision(numeric_gate_path: Path, dataloader_gate_path: Path) -> dict:
    numeric = json.loads(numeric_gate_path.read_text())
    dataloader = json.loads(dataloader_gate_path.read_text())
    if numeric.get("status") != "passed":
        raise ValueError("The mandatory fp32 numerical gate did not pass.")
    if numeric.get("results", {}).get("32-true", {}).get("status") != "passed":
        raise ValueError("The numerical report lacks a passing fp32 result.")
    precision = numeric.get("selected_precision")
    if precision not in {"32-true", "bf16-mixed"}:
        raise ValueError("The numerical gate selected an unsupported precision.")
    if precision == "bf16-mixed" and numeric.get("results", {}).get(
        "bf16-mixed", {}
    ).get("status") != "passed":
        raise ValueError("bf16 was selected without a passing bf16 result.")
    if dataloader.get("status") != "passed":
        raise ValueError("The DataLoader equivalence gate did not pass.")
    workers = dataloader.get("selected_workers")
    if workers not in {0, 2, 4, 8}:
        raise ValueError("The DataLoader gate selected an unregistered worker count.")
    selected = [
        row
        for row in dataloader.get("results", [])
        if row.get("workers") == workers and row.get("status") == "passed"
    ]
    anchor = [
        row
        for row in dataloader.get("results", [])
        if row.get("workers") == 0 and row.get("status") == "passed"
    ]
    if (
        len(selected) != 1
        or len(anchor) != 1
        or selected[0].get("order_sha256") != anchor[0].get("order_sha256")
    ):
        raise ValueError("Selected DataLoader workers are not order-equivalent to worker zero.")
    return {
        "precision": precision,
        "num_workers": int(workers),
        "numeric_gate_sha256": sha256_file(numeric_gate_path),
        "dataloader_gate_sha256": sha256_file(dataloader_gate_path),
    }


def build_command(
    *,
    selection: dict,
    runtime: dict,
    dataset: str,
    arm: str,
    seed: int,
    run_dir: Path,
    python: str = sys.executable,
) -> list[str]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported frozen dataset: {dataset}")
    if arm not in ARMS:
        raise ValueError(f"Unsupported frozen CytoBridge arm: {arm}")
    if dataset == "tahoe" and arm != "cytobridge_full":
        raise ValueError("Tahoe is registered for the full CytoBridge arm only.")
    if seed not in FINAL_SEEDS:
        raise ValueError(f"Final seed {seed} is outside the frozen five-seed set.")
    config_name, protocol = DATASETS[dataset]
    workers = int(runtime["num_workers"])
    overrides: dict[str, object] = {
        "seed": seed,
        "protocol": protocol,
        "run_name": run_dir.name,
        "run_metadata_path": str((run_dir / "run_metadata.json").resolve()),
        "ckpt.dirpath": str((run_dir / "ckpts").resolve()),
        "gradient_audit.output_path": str(
            (run_dir / "gradient_norms.jsonl").resolve()
        ),
        "trainer.max_epochs": int(selection["selected_epoch"]),
        "trainer.precision": runtime["precision"],
        "data.num_workers": workers,
        "data.persistent_workers": workers > 0,
        "data.prefetch_factor": 1 if workers > 0 else None,
        "data.multiprocessing_context": "spawn" if workers > 0 else None,
        **selection["selected_hyperparameters"],
        **ARMS[arm],
    }
    return [
        python,
        "train.py",
        f"--config-name=train/{config_name}",
        *(f"{key}={_hydra_value(value)}" for key, value in overrides.items()),
    ]


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--sweeps",
        type=Path,
        default=Path("../protocols/materialized_sweeps.json"),
    )
    parser.add_argument("--numeric-gate", type=Path, required=True)
    parser.add_argument("--dataloader-gate", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), default="cytobridge_full")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection = load_selection(args.selection, args.sweeps)
    runtime = load_runtime_decision(args.numeric_gate, args.dataloader_gate)
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to reuse frozen run directory: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    command = build_command(
        selection=selection,
        runtime=runtime,
        dataset=args.dataset,
        arm=args.arm,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    plan = {
        "schema_version": 1,
        "dataset": args.dataset,
        "arm": args.arm,
        "seed": args.seed,
        "selected_config_id": selection["selected_config_id"],
        "selected_epoch": selection["selected_epoch"],
        "selected_hyperparameters": selection["selected_hyperparameters"],
        "retuning_performed": False,
        "selection_sha256": sha256_file(args.selection),
        "numeric_gate_sha256": runtime["numeric_gate_sha256"],
        "dataloader_gate_sha256": runtime["dataloader_gate_sha256"],
        "precision": runtime["precision"],
        "num_workers": runtime["num_workers"],
        "command": command,
    }
    _write_json(args.run_dir / "frozen_execution.json", plan)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return

    log_path = args.run_dir / "train.log"
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            command,
            cwd=CODE_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    checkpoint_path = args.run_dir / "ckpts" / "last.ckpt"
    success = completed.returncode == 0 and checkpoint_path.is_file()
    status = {
        "schema_version": 1,
        "status": "complete" if success else "failed",
        "returncode": completed.returncode,
        "checkpoint": str(checkpoint_path) if checkpoint_path.is_file() else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "train_log_sha256": sha256_file(log_path),
        "automatic_retry": False,
    }
    _write_json(args.run_dir / "run.status", status)
    print(json.dumps(status, sort_keys=True))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
