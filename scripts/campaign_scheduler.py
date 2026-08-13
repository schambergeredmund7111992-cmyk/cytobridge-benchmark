"""Run one frozen campaign phase across every confirmed idle GPU without retries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from eval.artifacts import sha256_file

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
CORE_RUNNER = CODE_ROOT / "env/core_python.sh"
FINITE_LOSS = re.compile(
    r"(?:train/)?loss(?:_(?:step|epoch))?\s*[=:]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _jobs(manifest_path: Path, phase: str) -> list[dict]:
    payload = json.loads(manifest_path.read_text())
    jobs = [job for job in payload["jobs"] if job["phase"] == phase]
    if not jobs:
        raise ValueError(f"Campaign contains no jobs for phase {phase}.")
    pending = []
    for job in jobs:
        run_dir = RUN_ROOT / job["job_id"]
        if not run_dir.exists():
            pending.append(job)
            continue
        status_path = run_dir / "run.status"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if status.get("status") != "complete":
            raise FileExistsError(
                f"Phase {phase} contains an immutable incomplete run: {run_dir}"
            )
    return pending


def _preflight(path: Path, expected_host: str) -> list[int]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "passed"
        or payload.get("pi_confirmed") is not True
        or payload.get("host") != expected_host
    ):
        raise PermissionError(
            "Campaign scheduler requires confirmed preflight evidence for "
            f"{expected_host!r}."
        )
    eligible = [int(value) for value in payload.get("eligible_gpus", [])]
    if not eligible:
        raise ValueError("Confirmed preflight contains no eligible GPUs.")
    return eligible


def _first_finite_loss_line(job_id: str) -> str | None:
    log_path = RUN_ROOT / job_id / "run.log"
    if not log_path.is_file():
        return None
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 256 * 1024))
        text = handle.read().decode(errors="replace")
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        match = FINITE_LOSS.search(line)
        if match is not None:
            value = float(match.group(1))
            if value == value and abs(value) != float("inf"):
                return line[-800:]
    return None


def _launch(
    job: dict,
    manifest_path: Path,
    preflight_path: Path,
    expected_host: str,
    gpu: int | None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CYTOBRIDGE_LAUNCH_CONFIRMED"] = "1"
    env["CYTOBRIDGE_PREFLIGHT_PATH"] = str(preflight_path)
    env["CYTOBRIDGE_EXPECTED_HOST"] = expected_host
    env.setdefault(
        "CYTOBRIDGE_SOURCE_MANIFEST",
        str(EXPERIMENT_ROOT / "gates/source_tree_manifest.json"),
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [
        str(CORE_RUNNER),
        "-m",
        "scripts.run_campaign_job",
        "--manifest",
        str(manifest_path),
        "--job-id",
        job["job_id"],
    ]
    return subprocess.Popen(
        command,
        cwd=CODE_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_phase(
    *,
    manifest_path: Path,
    phase: str,
    preflight_path: Path,
    expected_host: str,
    cpu_parallelism: int,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite scheduler report: {output_path}")
    jobs = _jobs(manifest_path, phase)
    eligible_gpus = _preflight(preflight_path, expected_host)
    learned = [job for job in jobs if job["learned"]]
    cpu = [job for job in jobs if not job["learned"]]
    free_gpus = list(eligible_gpus)
    running: dict[int, tuple[dict, int | None, subprocess.Popen]] = {}
    completed = []
    failed = []
    finite_loss_jobs: set[str] = set()
    stop_launching = False
    print(
        json.dumps(
            {
                "phase": phase,
                "learned_jobs": len(learned),
                "cpu_jobs": len(cpu),
                "eligible_gpus": eligible_gpus,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not jobs:
        report = {
            "schema_version": 1,
            "status": "complete",
            "phase": phase,
            "manifest_sha256": sha256_file(manifest_path),
            "preflight_sha256": sha256_file(preflight_path),
            "eligible_gpus": eligible_gpus,
            "completed": [],
            "failed": [],
            "not_launched": [],
            "automatic_retries": 0,
            "all_jobs_already_complete": True,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    while learned or cpu or running:
        if not stop_launching:
            while learned and free_gpus:
                job = learned.pop(0)
                gpu = free_gpus.pop(0)
                process = _launch(
                    job, manifest_path, preflight_path, expected_host, gpu
                )
                running[process.pid] = (job, gpu, process)
                print(f"LAUNCHED job={job['job_id']} gpu={gpu}", flush=True)
            running_cpu = sum(gpu is None for _, gpu, _ in running.values())
            while cpu and running_cpu < cpu_parallelism:
                job = cpu.pop(0)
                process = _launch(
                    job, manifest_path, preflight_path, expected_host, None
                )
                running[process.pid] = (job, None, process)
                running_cpu += 1
                print(f"LAUNCHED job={job['job_id']} cpu=true", flush=True)

        finished = []
        for pid, (job, gpu, process) in running.items():
            if job["learned"] and job["job_id"] not in finite_loss_jobs:
                finite_line = _first_finite_loss_line(job["job_id"])
                if finite_line is not None:
                    finite_loss_jobs.add(job["job_id"])
                    print(
                        f"FIRST_FINITE_LOSS job={job['job_id']} line={finite_line}",
                        flush=True,
                    )
            returncode = process.poll()
            if returncode is None:
                continue
            finished.append(pid)
            if gpu is not None:
                free_gpus.append(gpu)
                free_gpus.sort()
            status_path = RUN_ROOT / job["job_id"] / "run.status"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            record = {
                "job_id": job["job_id"],
                "returncode": returncode,
                "status": status.get("status", "missing"),
                "status_sha256": (
                    sha256_file(status_path) if status_path.is_file() else None
                ),
            }
            if returncode == 0 and status.get("status") == "complete":
                completed.append(record)
                print(f"COMPLETE job={job['job_id']}", flush=True)
            else:
                failed.append(record)
                print(f"FAILED job={job['job_id']} (will skip, continuing phase)", flush=True)
        for pid in finished:
            running.pop(pid)
        if stop_launching and not running:
            break
        if running and not finished:
            time.sleep(5)

    report = {
        "schema_version": 1,
        "status": "complete" if not failed and not learned and not cpu else "failed",
        "phase": phase,
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "expected_host": expected_host,
        "eligible_gpus": eligible_gpus,
        "finite_loss_jobs": sorted(finite_loss_jobs),
        "completed": completed,
        "failed": failed,
        "not_launched": [job["job_id"] for job in [*learned, *cpu]],
        "automatic_retries": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("P1", "P2", "P3"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--expected-host",
        default=os.environ.get("CYTOBRIDGE_EXPECTED_HOST", "cityu"),
    )
    parser.add_argument("--cpu-parallelism", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_parallelism < 1:
        raise ValueError("cpu-parallelism must be positive.")
    report = run_phase(
        manifest_path=args.manifest.resolve(),
        phase=args.phase,
        preflight_path=args.preflight.resolve(),
        expected_host=args.expected_host,
        cpu_parallelism=args.cpu_parallelism,
        output_path=args.out.resolve(),
    )
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
