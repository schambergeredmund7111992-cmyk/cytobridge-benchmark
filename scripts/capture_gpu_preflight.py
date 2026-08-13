"""Capture a fresh GPU/RAM/disk inventory and conservative campaign estimate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cytobridge.model import CytoBridge, CytoBridgeConfig

CODE_ROOT = Path(__file__).resolve().parents[1]


def _gpu_rows() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"Could not parse nvidia-smi row: {line}")
        index, name, utilization, memory_used, memory_total = fields
        row = {
            "index": int(index),
            "name": name,
            "utilization_percent": int(utilization),
            "memory_used_mib": int(memory_used),
            "memory_total_mib": int(memory_total),
        }
        row["eligible"] = (
            row["utilization_percent"] < 5 and row["memory_used_mib"] < 1024
        )
        rows.append(row)
    return rows


def _parameter_count() -> int:
    config = yaml.safe_load(
        (CODE_ROOT / "configs/train/accept_base.yaml").read_text()
    )["model"]
    model = CytoBridge(CytoBridgeConfig(**config))
    return sum(parameter.numel() for parameter in model.parameters())


def capture(output_path: Path, *, host_label: str = "cityu") -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite GPU preflight draft: {output_path}")
    gpu_rows = _gpu_rows()
    eligible = [row["index"] for row in gpu_rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No GPU satisfies utilization<5% and memory<1 GiB.")
    parameters = _parameter_count()
    available_memory_bytes = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf(
        "SC_PAGE_SIZE"
    )
    disk = shutil.disk_usage(CODE_ROOT)
    payload = {
        "schema_version": 1,
        "status": "draft",
        "host": host_label,
        "host_node": platform.node(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpus": gpu_rows,
        "eligible_gpus": eligible,
        "resource_policy": "use every GPU with utilization<5% and memory<1GiB",
        "cytobridge_trainable_parameters": parameters,
        "estimate": {
            "vram_gib_per_learned_job": [18, 28],
            "cpu_ram_gib_per_learned_job": [30, 60],
            "campaign_checkpoint_and_result_disk_gib": [90, 140],
            "total_gpu_hours": [140, 260],
            "wall_hours_if_all_eligible_remain_available": [
                round(140 / len(eligible), 1),
                round(260 / len(eligible), 1),
            ],
            "basis": (
                "frozen 79 learned jobs; fp32-safe upper bound; one GPU per job; "
                "external 201/500-epoch budgets included"
            ),
        },
        "host_resources": {
            "ram_available_gib": round(available_memory_bytes / 2**30, 1),
            "disk_free_gib": round(disk.free / 2**30, 1),
        },
        "pi_confirmed": False,
        "training_launched": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--host-label",
        default=os.environ.get("CYTOBRIDGE_HOST_LABEL", "cityu"),
        help="Execution target recorded in the approval evidence (for example cityu or autodl).",
    )
    args = parser.parse_args()
    if not args.host_label or any(character.isspace() for character in args.host_label):
        raise ValueError("host-label must be a non-empty token without whitespace.")
    result = capture(args.out, host_label=args.host_label)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
