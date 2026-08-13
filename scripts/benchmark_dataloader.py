"""Choose a safe DataLoader worker count using label-free, order-equivalent timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from cytobridge.data import CytoBridgeDataModule


def _tail_text(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _load_config(config_name: str, workers: int) -> dict:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=config_name)
    values = OmegaConf.to_container(cfg.data, resolve=True)
    values["num_workers"] = workers
    values["persistent_workers"] = workers > 0
    values["prefetch_factor"] = 1 if workers > 0 else None
    values["multiprocessing_context"] = "spawn" if workers > 0 else None
    return values


def _run_worker(config_name: str, workers: int, batches: int, output: Path) -> None:
    data_module = CytoBridgeDataModule(**_load_config(config_name, workers))
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    digest = hashlib.sha256()
    rows = 0
    completed_batches = 0
    started = time.perf_counter()
    iterator = iter(loader)
    try:
        for batch_index, batch in enumerate(iterator):
            identity = {
                "anchor_indices": batch["anchor_indices"],
                "hard_negative_indices": batch["hard_neg_indices"],
            }
            digest.update(json.dumps(identity, sort_keys=True).encode())
            rows += len(batch["anchor_indices"])
            completed_batches = batch_index + 1
            if batch_index + 1 >= batches:
                break
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    seconds = time.perf_counter() - started
    result = {
        "status": "passed",
        "workers": workers,
        "batches": completed_batches,
        "rows": rows,
        "seconds": seconds,
        "rows_per_second": rows / seconds,
        "order_sha256": digest.hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _run_parent(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite DataLoader benchmark: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for workers in args.workers:
        child_out = args.out.with_name(f"{args.out.stem}.workers-{workers}.json")
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_dataloader",
            "--worker-run",
            "--config-name",
            args.config_name,
            "--worker-count",
            str(workers),
            "--batches",
            str(args.batches),
            "--out",
            str(child_out),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            if completed.returncode == 0 and child_out.is_file():
                result = json.loads(child_out.read_text())
            else:
                result = {
                    "status": "failed",
                    "workers": workers,
                    "returncode": completed.returncode,
                    "stdout": _tail_text(completed.stdout),
                    "stderr": _tail_text(completed.stderr),
                }
        except subprocess.TimeoutExpired as exc:
            result = {
                "status": "timeout",
                "workers": workers,
                "timeout_seconds": args.timeout,
                "stdout": _tail_text(exc.stdout),
                "stderr": _tail_text(exc.stderr),
            }
        results.append(result)
    reference = next(
        (item for item in results if item["workers"] == 0 and item["status"] == "passed"),
        None,
    )
    eligible = []
    if reference is not None:
        eligible = [
            item
            for item in results
            if item["status"] == "passed"
            and item.get("order_sha256") == reference["order_sha256"]
        ]
    selected = max(eligible, key=lambda item: item["rows_per_second"], default=None)
    report = {
        "schema_version": 1,
        "config_name": args.config_name,
        "selection_rule": "fastest order-equivalent successful candidate",
        "results": results,
        "selected_workers": selected["workers"] if selected else None,
        "status": "passed" if selected else "failed",
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "workers": report["selected_workers"]}))
    return 0 if selected else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train/accept_base")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-run", action="store_true")
    parser.add_argument("--worker-count", type=int, default=0)
    args = parser.parse_args()
    if args.worker_run:
        _run_worker(args.config_name, args.worker_count, args.batches, args.out)
        return
    raise SystemExit(_run_parent(args))


if __name__ == "__main__":
    main()
