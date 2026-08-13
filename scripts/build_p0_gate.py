"""Build the fail-closed P0 gate from data, cache, runtime, and baseline evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.artifacts import sha256_file
from scripts.control_calibration import calibrate_controls
from scripts.run_frozen_cytobridge import load_runtime_decision

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
GATE_ROOT = EXPERIMENT_ROOT / "gates"


def _external_state() -> tuple[Path, dict]:
    root = Path(
        os.environ.get(
            "CYTOBRIDGE_EXTERNAL_ROOT",
            f"/tmp/{os.environ.get('USER', 'unknown')}/cytobridge-accept-external",
        )
    )
    path = root / "state/checkout_verification.json"
    payload = json.loads(path.read_text())
    checkouts = payload.get("checkouts", [])
    if (
        len(checkouts) != 3
        or any(row.get("clean") is not True for row in checkouts)
        or payload.get("chemcpa_initialization") != "from_scratch"
        or payload.get("pretrained_weights_used") is not False
    ):
        raise ValueError("External checkout verification did not pass.")
    return path, payload


def _check_external_inputs() -> dict[str, str]:
    required = []
    for protocol, modes in (
        ("drug_disjoint_v2", ("selection", "final_refit")),
        ("scaffold_disjoint_v2", ("final_refit",)),
    ):
        base = CODE_ROOT / "data/processed/external" / protocol
        for mode in modes:
            export = base / f"{mode}.h5ad"
            export_manifest = export.with_suffix(".manifest.json")
            rdkit = base / f"{mode}_chemcpa_rdkit.parquet"
            rdkit_manifest = rdkit.with_suffix(".manifest.json")
            required.extend((export, export_manifest, rdkit, rdkit_manifest))
            export_payload = json.loads(export_manifest.read_text())
            rdkit_payload = json.loads(rdkit_manifest.read_text())
            if mode == "selection":
                if (
                    export_payload.get("test_responses_included") is not False
                    or export_payload.get("included_benchmark_splits")
                    != ["train", "validation"]
                ):
                    raise ValueError("External selection export contains test rows.")
            elif export_payload.get("train_validation_refit_union") is not True:
                raise ValueError("External final export did not combine train+validation.")
            if (
                rdkit_payload.get("export_sha256") != sha256_file(export)
                or rdkit_payload.get("output_sha256") != sha256_file(rdkit)
                or rdkit_payload.get("validation_or_test_response_used") is not False
            ):
                raise ValueError("chemCPA descriptor provenance differs from its export.")
    return {str(path.relative_to(CODE_ROOT)): sha256_file(path) for path in required}


def _handoff_readiness(scgpt_ckpt_dir: Path) -> tuple[dict, str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "handoff/check_readiness.py"),
        "--code-dir",
        str(CODE_ROOT),
        "--scgpt-ckpt-dir",
        str(scgpt_ckpt_dir),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    if completed.returncode or report.get("training_ready") is not True:
        raise ValueError("Handoff readiness did not pass every required check.")
    return report, " ".join(command)


def build_p0_gate(
    *,
    scgpt_ckpt_dir: Path,
    numeric_gate_path: Path,
    dataloader_gate_path: Path,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite P0 gate: {output_path}")
    readiness, readiness_command = _handoff_readiness(scgpt_ckpt_dir)
    runtime = load_runtime_decision(numeric_gate_path, dataloader_gate_path)
    checkout_path, _ = _external_state()
    external_inputs = _check_external_inputs()
    controls = {
        protocol: calibrate_controls(
            CODE_ROOT / "data/processed/sciplex_accept" / protocol
        )
        for protocol in ("drug_disjoint_v2", "scaffold_disjoint_v2")
    }
    payload = {
        "schema_version": 1,
        "status": "passed",
        "protocol_version": "1.5.0",
        "test_targets_opened": False,
        "readiness_command": readiness_command,
        "readiness_training_ready": readiness["training_ready"],
        "precision": runtime["precision"],
        "num_workers": runtime["num_workers"],
        "control_calibration": controls,
        "source_hashes": {
            "numeric_gate": sha256_file(numeric_gate_path),
            "dataloader_gate": sha256_file(dataloader_gate_path),
            "external_checkout_verification": sha256_file(checkout_path),
            **external_inputs,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scgpt-ckpt-dir", type=Path, required=True)
    parser.add_argument(
        "--numeric-gate",
        type=Path,
        default=GATE_ROOT / "numeric_first_batch.json",
    )
    parser.add_argument(
        "--dataloader-gate",
        type=Path,
        default=GATE_ROOT / "dataloader.json",
    )
    parser.add_argument("--out", type=Path, default=GATE_ROOT / "p0_ready.json")
    args = parser.parse_args()
    result = build_p0_gate(
        scgpt_ckpt_dir=args.scgpt_ckpt_dir,
        numeric_gate_path=args.numeric_gate,
        dataloader_gate_path=args.dataloader_gate,
        output_path=args.out,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
