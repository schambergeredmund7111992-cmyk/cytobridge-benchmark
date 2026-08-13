"""Execute one frozen campaign job with fail-closed gates and immutable outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval.artifacts import sha256_file
from scripts.run_frozen_cytobridge import (
    build_command,
    load_runtime_decision,
    load_selection,
)
from scripts.source_tree_manifest import build_manifest

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
PROTOCOL_ROOT = PROJECT_ROOT / "protocols"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
GATE_ROOT = EXPERIMENT_ROOT / "gates"
SELECTION_ROOT = EXPERIMENT_ROOT / "selections"
SWEEPS = PROTOCOL_ROOT / "materialized_sweeps.json"
NUMERIC_GATE = GATE_ROOT / "numeric_first_batch.json"
DATALOADER_GATE = GATE_ROOT / "dataloader.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True, default=str)
        + "\n"
    )
    temporary.replace(path)


def _load_job(manifest_path: Path, job_id: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    matches = [job for job in manifest["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one campaign job {job_id!r}; observed {len(matches)}.")
    job = matches[0]
    if job["command"][-1] != job_id:
        raise ValueError("Campaign job command and selected job identifier differ.")
    return manifest, job


def _require_dependencies(job: dict) -> dict[str, str]:
    hashes = {}
    for dependency in job["depends_on"]:
        if not str(dependency).startswith("gate:"):
            raise ValueError(f"Unsupported non-gate dependency: {dependency}")
        gate_name = str(dependency).removeprefix("gate:")
        gate_path = GATE_ROOT / f"{gate_name}.json"
        if not gate_path.is_file():
            raise FileNotFoundError(f"Required campaign gate is missing: {gate_path}")
        payload = json.loads(gate_path.read_text())
        if payload.get("status") not in {"passed", "complete"}:
            raise ValueError(f"Campaign gate did not pass: {gate_path}")
        hashes[gate_name] = sha256_file(gate_path)
    return hashes


def _require_learned_launch_authorization(job: dict) -> dict[str, str]:
    if not job["learned"]:
        return {}
    if os.environ.get("CYTOBRIDGE_LAUNCH_CONFIRMED") != "1":
        raise PermissionError(
            "Learned campaign launch requires the explicitly confirmed preflight flag."
        )
    preflight_path = Path(
        os.environ.get("CYTOBRIDGE_PREFLIGHT_PATH", GATE_ROOT / "gpu_preflight.json")
    )
    preflight = json.loads(preflight_path.read_text())
    expected_host = os.environ.get("CYTOBRIDGE_EXPECTED_HOST", "cityu")
    if (
        preflight.get("status") != "passed"
        or preflight.get("pi_confirmed") is not True
        or preflight.get("host") != expected_host
    ):
        raise PermissionError(
            "GPU preflight evidence is missing explicit PI confirmation for "
            f"{expected_host!r}."
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if len(devices) != 1 or not devices[0].isdigit():
        raise ValueError("Each learned campaign job must expose exactly one physical GPU.")
    _gpu_idle_timeout = 120
    _gpu_poll_interval = 5
    _gpu_deadline = time.time() + _gpu_idle_timeout
    while True:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={devices[0]}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        fields = [int(value.strip()) for value in query.split(",")]
        if len(fields) == 3 and fields[0] < 5 and fields[1] < 1024:
            break
        if time.time() >= _gpu_deadline:
            raise RuntimeError(
                f"Assigned GPU {devices[0]} did not become idle within {_gpu_idle_timeout}s: {query}"
            )
        print(
            f"GPU {devices[0]} busy (util={fields[0]}%, mem={fields[1]}MiB), "
            f"waiting {_gpu_poll_interval}s...",
            flush=True,
        )
        time.sleep(_gpu_poll_interval)
    if devices[0] not in {str(value) for value in preflight.get("eligible_gpus", [])}:
        raise ValueError("Assigned GPU was not eligible in the confirmed preflight.")
    return {"gpu_preflight": sha256_file(preflight_path)}


def _require_source_tree() -> dict[str, str]:
    manifest_path = Path(
        os.environ.get(
            "CYTOBRIDGE_SOURCE_MANIFEST",
            GATE_ROOT / "source_tree_manifest.json",
        )
    )
    expected = json.loads(manifest_path.read_text())
    observed = build_manifest(PROJECT_ROOT, exclude_paths=(manifest_path,))
    if expected != observed:
        raise ValueError(
            "Source tree differs from the immutable campaign manifest: "
            f"{observed['tree_sha256']} != {expected.get('tree_sha256')}"
        )
    return {
        "source_tree_manifest": sha256_file(manifest_path),
        "source_tree_sha256": str(expected["tree_sha256"]),
    }


def _dataset_paths(dataset: str) -> dict[str, Any]:
    if dataset == "sciplex_drug_disjoint_v2":
        protocol = "drug_disjoint_v2"
        root = CODE_ROOT / "data/processed/sciplex_accept" / protocol
        prefix = "sciplex"
        cache = CODE_ROOT / "data/cache/sciplex_accept" / protocol
        smiles = root / "eligible_compounds.csv"
        dataset_name = "sci-plex"
    elif dataset == "sciplex_scaffold_disjoint_v2":
        protocol = "scaffold_disjoint_v2"
        root = CODE_ROOT / "data/processed/sciplex_accept" / protocol
        prefix = "sciplex"
        cache = CODE_ROOT / "data/cache/sciplex_accept" / protocol
        smiles = root / "eligible_compounds.csv"
        dataset_name = "sci-plex"
    elif dataset == "tahoe":
        protocol = "tahoe"
        root = CODE_ROOT / "data/processed/tahoe_accept"
        prefix = "tahoe"
        cache = CODE_ROOT / "data/cache/tahoe_accept"
        smiles = root / "drug_smiles.csv"
        dataset_name = "tahoe-100m"
    else:
        raise ValueError(f"Unsupported campaign dataset: {dataset}")
    split = root / "splits"
    return {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "protocol": protocol,
        "root": root,
        "split": split,
        "prefix": prefix,
        "cache": cache,
        "smiles": smiles,
        "split_manifest": root / "split_manifest.json",
        "gene_panels": split / "training_gene_panels.json",
        "train_targets": split / "train_targets.npz",
        "train_metadata": split / "train_targets_metadata.csv",
        "validation_targets": split / "val_targets.npz",
        "validation_metadata": split / "val_targets_metadata.csv",
        "test_targets": split / "test_targets.npz",
        "test_metadata": split / "test_targets_metadata.csv",
        "train_manifest": split / f"{prefix}_train.parquet",
        "validation_manifest": split / f"{prefix}_val.parquet",
        "test_manifest": split / f"{prefix}_test.parquet",
        "train_treated": split / f"{prefix}_train_treated_counts.npy",
        "validation_treated": split / f"{prefix}_val_treated_counts.npy",
        "test_treated": split / f"{prefix}_test_treated_counts.npy",
        "test_input_control": split / f"{prefix}_test_input_control_counts.npy",
        "test_truth_control": split / f"{prefix}_test_truth_control_counts.npy",
        "test_gsea": split / f"{prefix}_test_pathway_gsea.npy",
        "cell_emb": cache / "scgpt_emb.npy",
        "drug_emb": cache / "molformer_emb.npz",
    }


def _run(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_handle:
        completed = subprocess.run(
            command,
            cwd=CODE_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _runtime() -> dict:
    return load_runtime_decision(NUMERIC_GATE, DATALOADER_GATE)


def _screen_cytobridge(job: dict, run_dir: Path, log_path: Path) -> dict:
    runtime = _runtime()
    sweeps = json.loads(SWEEPS.read_text())
    matches = [
        row for row in sweeps["cytobridge"] if row["config_id"] == job["config_id"]
    ]
    if len(matches) != 1 or int(job["seed"]) != 42:
        raise ValueError("CytoBridge screening job differs from the registered sweep.")
    trial = matches[0]
    workers = int(runtime["num_workers"])
    overrides = {
        "seed": 42,
        "protocol": "drug_disjoint_v2",
        "run_name": job["job_id"],
        "run_metadata_path": str((run_dir / "run_metadata.json").resolve()),
        "metric.selection_state_path": str(
            (run_dir / "selection_state.json").resolve()
        ),
        "gradient_audit.output_path": str(
            (run_dir / "gradient_norms.jsonl").resolve()
        ),
        "ckpt.dirpath": str((run_dir / "ckpts").resolve()),
        "trainer.precision": runtime["precision"],
        "data.num_workers": workers,
        "data.persistent_workers": str(workers > 0).lower(),
        "data.prefetch_factor": 1 if workers > 0 else "null",
        "data.multiprocessing_context": "spawn" if workers > 0 else "null",
        "loss.lam_recon": trial["loss.lam_recon"],
        "loss.lam_drugspec": trial["loss.lam_drugspec"],
    }
    command = [
        sys.executable,
        "train.py",
        "--config-name=train/accept_base",
        *(f"{key}={value}" for key, value in overrides.items()),
    ]
    _run(command, log_path)
    required = [
        run_dir / "selection_state.json",
        run_dir / "run_metadata.json",
        run_dir / "ckpts/best.ckpt",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("CytoBridge screen completed without frozen selection outputs.")
    return {
        "command": command,
        "output_hashes": {path.name: sha256_file(path) for path in required},
    }


def _external_environment() -> tuple[str, dict[str, Path], dict[str, str]]:
    external_root = Path(
        os.environ.get(
            "CYTOBRIDGE_EXTERNAL_ROOT",
            f"/tmp/{os.environ.get('USER', 'unknown')}/cytobridge-accept-external",
        )
    )
    sources = {
        "chemcpa": external_root / "sources/chemCPA",
        "biolord": external_root / "sources/biolord",
        "biolord_repro": external_root / "sources/biolord_reproducibility",
    }
    python = os.environ.get(
        "CYTOBRIDGE_EXTERNAL_PYTHON",
        str(Path.home() / "anaconda3/envs/cytobridge-accept-external/bin/python"),
    )
    if not Path(python).is_file() or not all(path.is_dir() for path in sources.values()):
        raise FileNotFoundError("Frozen external baseline environment is incomplete.")
    env = os.environ.copy()
    external_prefix = str(Path(python).resolve().parents[1])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [external_prefix + "/lib", env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE_ROOT), str(sources["chemcpa"]), str(sources["biolord"] / "src")]
    )
    return python, sources, env


def _external_export(protocol: str, mode: str) -> tuple[Path, Path]:
    base = CODE_ROOT / "data/processed/external" / protocol
    export = base / f"{mode}.h5ad"
    rdkit = base / f"{mode}_chemcpa_rdkit.parquet"
    return export, rdkit


def _run_external(
    job: dict,
    run_dir: Path,
    log_path: Path,
    paths: dict,
    *,
    mode: str,
    config_id: str,
    epochs: int,
) -> tuple[Path, dict]:
    python, sources, env = _external_environment()
    export_name = "selection" if mode == "selection" else "final_refit"
    export, rdkit = _external_export(paths["protocol"], export_name)
    output = run_dir / "external_output"
    module = (
        "external.chemcpa_runtime"
        if job["model"] == "chemcpa"
        else "external.biolord_runtime"
    )
    command = [python, "-m", module]
    if job["model"] == "chemcpa":
        command.extend(
            [
                "--chemcpa-source",
                str(sources["chemcpa"]),
                "--rdkit-embeddings",
                str(rdkit),
            ]
        )
    else:
        command.extend(
            [
                "--biolord-source",
                str(sources["biolord"]),
                "--biolord-repro",
                str(sources["biolord_repro"]),
            ]
        )
    benchmark_manifest = (
        paths["validation_manifest"] if mode == "selection" else paths["test_manifest"]
    )
    treated = paths["validation_treated"] if mode == "selection" else paths["test_treated"]
    truth_control = (
        paths["split"] / f"{paths['prefix']}_val_truth_control_counts.npy"
        if mode == "selection"
        else paths["test_truth_control"]
    )
    command.extend(
        [
            "--export",
            str(export),
            "--sweeps",
            str(SWEEPS),
            "--config-id",
            config_id,
            "--seed",
            str(job["seed"]),
            "--mode",
            mode,
            "--epochs",
            str(epochs),
            "--manifest",
            str(benchmark_manifest),
            "--treated-counts",
            str(treated),
            "--truth-control-counts",
            str(truth_control),
            "--gene-panels",
            str(paths["gene_panels"]),
            "--out",
            str(output),
        ]
    )
    _run(command, log_path, env=env)
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError("External baseline completed without result.json.")
    return output, {"command": command, "result_sha256": sha256_file(result_path)}


def _screen_external(job: dict, run_dir: Path, log_path: Path) -> dict:
    paths = _dataset_paths("sciplex_drug_disjoint_v2")
    epochs = 201 if job["model"] == "chemcpa" else 500
    _, details = _run_external(
        job,
        run_dir,
        log_path,
        paths,
        mode="selection",
        config_id=job["config_id"],
        epochs=epochs,
    )
    return details


def _screen_ridge(run_dir: Path, log_path: Path) -> dict:
    paths = _dataset_paths("sciplex_drug_disjoint_v2")
    selection = run_dir / "selection.json"
    trials = run_dir / "trials.csv"
    command = [
        sys.executable,
        "-m",
        "eval.baselines.ridge_pseudobulk",
        "select",
        "--train-targets",
        str(paths["train_targets"]),
        "--train-metadata",
        str(paths["train_metadata"]),
        "--validation-targets",
        str(paths["validation_targets"]),
        "--validation-metadata",
        str(paths["validation_metadata"]),
        "--smiles-csv",
        str(paths["smiles"]),
        "--gene-panels",
        str(paths["gene_panels"]),
        "--out-selection",
        str(selection),
        "--out-trials",
        str(trials),
    ]
    _run(command, log_path)
    return {
        "command": command,
        "selection_sha256": sha256_file(selection),
        "trials_sha256": sha256_file(trials),
    }


def _selection_path(model: str) -> Path:
    path = SELECTION_ROOT / f"{model}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen P1 model selection is missing: {path}")
    return path


def _predict_cytobridge(
    checkpoint: Path,
    paths: dict,
    runtime: dict,
    output: Path,
    log_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eval.predict_cytobridge",
        "--ckpt",
        str(checkpoint),
        "--manifest",
        str(paths["test_manifest"]),
        "--cell-emb",
        str(paths["cell_emb"]),
        "--drug-emb",
        str(paths["drug_emb"]),
        "--treated-counts",
        str(paths["test_treated"]),
        "--input-control-counts",
        str(paths["test_input_control"]),
        "--truth-control-counts",
        str(paths["test_truth_control"]),
        "--pathway-gsea",
        str(paths["test_gsea"]),
        "--frozen-targets",
        str(paths["test_targets"]),
        "--frozen-metadata",
        str(paths["test_metadata"]),
        "--num-workers",
        str(runtime["num_workers"]),
        "--out",
        str(output),
    ]
    _run(command, log_path)
    return command


def _frozen_cytobridge_arm(model: str) -> str:
    """Resolve the public transfer label to the registered full-model arm."""
    return "cytobridge_full" if model == "cytobridge" else model


def _run_final_cytobridge(job: dict, run_dir: Path, log_path: Path, paths: dict) -> dict:
    selection_path = _selection_path("cytobridge")
    selection = load_selection(selection_path, SWEEPS)
    runtime = _runtime()
    training_dir = run_dir / "training"
    command = build_command(
        selection=selection,
        runtime=runtime,
        dataset=job["dataset"],
        arm=_frozen_cytobridge_arm(job["model"]),
        seed=int(job["seed"]),
        run_dir=training_dir,
    )
    runner = [
        sys.executable,
        "-m",
        "scripts.run_frozen_cytobridge",
        "--selection",
        str(selection_path),
        "--sweeps",
        str(SWEEPS),
        "--numeric-gate",
        str(NUMERIC_GATE),
        "--dataloader-gate",
        str(DATALOADER_GATE),
        "--dataset",
        job["dataset"],
        "--arm",
        _frozen_cytobridge_arm(job["model"]),
        "--seed",
        str(job["seed"]),
        "--run-dir",
        str(training_dir),
    ]
    _run(runner, log_path)
    checkpoint = training_dir / "ckpts/last.ckpt"
    predictions = run_dir / "predictions.npz"
    prediction_command = _predict_cytobridge(
        checkpoint, paths, runtime, predictions, log_path
    )
    return {
        "training_command": command,
        "runner_command": runner,
        "prediction_command": prediction_command,
        "predictions": predictions,
        "config": training_dir / "frozen_execution.json",
        "checkpoint": checkpoint,
        "extra_sources": {"selection": selection_path},
    }


def _run_reference(job: dict, run_dir: Path, log_path: Path, paths: dict) -> dict:
    kind = job["model"]
    seed = int(job["seed"] if job["seed"] is not None else 0)
    predictions = run_dir / "predictions.npz"
    command = [
        sys.executable,
        "-m",
        "eval.reference_controls",
        "--kind",
        kind,
        "--eval-targets",
        str(paths["test_targets"]),
        "--eval-metadata",
        str(paths["test_metadata"]),
        "--seed",
        str(seed),
        "--out",
        str(predictions),
    ]
    if kind == "mean":
        command.extend(
            [
                "--train-targets",
                str(paths["train_targets"]),
                "--train-metadata",
                str(paths["train_metadata"]),
            ]
        )
    _run(command, log_path)
    config = run_dir / "model_config.json"
    _write_json(
        config,
        {
            "model": kind,
            "seed": seed,
            "training_targets_used": kind == "mean",
            "test_targets_used_only_to_define_control": kind in {"random", "oracle"},
        },
    )
    return {
        "command": command,
        "predictions": predictions,
        "config": config,
        "checkpoint": None,
        "extra_sources": {},
    }


def _run_final_ridge(job: dict, run_dir: Path, log_path: Path, paths: dict) -> dict:
    if job["dataset"] == "tahoe":
        selection = run_dir / "selection.json"
        trials = run_dir / "trials.csv"
        select_command = [
            sys.executable,
            "-m",
            "eval.baselines.ridge_pseudobulk",
            "select",
            "--train-targets",
            str(paths["train_targets"]),
            "--train-metadata",
            str(paths["train_metadata"]),
            "--validation-targets",
            str(paths["validation_targets"]),
            "--validation-metadata",
            str(paths["validation_metadata"]),
            "--smiles-csv",
            str(paths["smiles"]),
            "--gene-panels",
            str(paths["gene_panels"]),
            "--out-selection",
            str(selection),
            "--out-trials",
            str(trials),
        ]
        _run(select_command, log_path)
    else:
        selection = _selection_path("ridge")
        select_command = None
    predictions = run_dir / "predictions.npz"
    command = [
        sys.executable,
        "-m",
        "eval.baselines.ridge_pseudobulk",
        "refit-predict",
        "--selection",
        str(selection),
        "--train-targets",
        str(paths["train_targets"]),
        "--train-metadata",
        str(paths["train_metadata"]),
        "--validation-targets",
        str(paths["validation_targets"]),
        "--validation-metadata",
        str(paths["validation_metadata"]),
        "--test-metadata",
        str(paths["test_metadata"]),
        "--smiles-csv",
        str(paths["smiles"]),
        "--out-predictions",
        str(predictions),
    ]
    _run(command, log_path)
    return {
        "selection_command": select_command,
        "command": command,
        "predictions": predictions,
        "config": selection,
        "checkpoint": None,
        "extra_sources": {"ridge_selection": selection},
    }


def _run_final_external(job: dict, run_dir: Path, log_path: Path, paths: dict) -> dict:
    selection_path = _selection_path(job["model"])
    selection = json.loads(selection_path.read_text())
    if selection.get("reuse_contract", {}).get("retuning_allowed") is not False:
        raise ValueError("External final job lacks a no-retuning selection contract.")
    output, details = _run_external(
        job,
        run_dir,
        log_path,
        paths,
        mode="final_refit",
        config_id=selection["selected_config_id"],
        epochs=int(selection["selected_epoch"]),
    )
    predictions = run_dir / "predictions.npz"
    command = [
        sys.executable,
        "-m",
        "eval.import_external_predictions",
        "--predicted-log1p",
        str(output / "predictions.npz"),
        "--eval-manifest",
        str(paths["test_manifest"]),
        "--true-treated",
        str(paths["test_treated"]),
        "--truth-control",
        str(paths["test_truth_control"]),
        "--frozen-targets",
        str(paths["test_targets"]),
        "--frozen-metadata",
        str(paths["test_metadata"]),
        "--out",
        str(predictions),
    ]
    _run(command, log_path)
    return {
        **details,
        "import_command": command,
        "predictions": predictions,
        "config": output / "result.json",
        "checkpoint": output / "model.pt",
        "extra_sources": {"selection": selection_path},
    }


def _package_and_score(
    job: dict,
    run_dir: Path,
    log_path: Path,
    paths: dict,
    result: dict,
) -> dict:
    artifact = run_dir / "artifact"
    scored = run_dir / "scored"
    seed = int(job["seed"] if job["seed"] is not None else 0)
    if job["model"] == "chemcpa":
        model_label = "chemCPA (from scratch)"
    elif job["model"] in {"cytobridge", "cytobridge_full"}:
        model_label = "cytobridge"
    else:
        model_label = job["model"]
    command = [
        sys.executable,
        "-m",
        "eval.package_artifact",
        "--predictions",
        str(result["predictions"]),
        "--targets",
        str(paths["test_targets"]),
        "--metadata",
        str(paths["test_metadata"]),
        "--gene-panels",
        str(paths["gene_panels"]),
        "--split-manifest",
        str(paths["split_manifest"]),
        "--config",
        str(result["config"]),
        "--out",
        str(artifact),
        "--model",
        model_label,
        "--seed",
        str(seed),
    ]
    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    for name, path in sorted(result.get("extra_sources", {}).items()):
        command.extend(["--source", f"{name}={path}"])
    _run(command, log_path)
    score_command = [
        sys.executable,
        "-m",
        "eval.run_benchmark",
        "--artifact",
        str(artifact),
        "--gene-panels",
        str(paths["gene_panels"]),
        "--out",
        str(scored),
        "--n-boot",
        "10000",
        "--n-permutations",
        "10000",
    ]
    _run(score_command, log_path)
    return {
        "package_command": command,
        "score_command": score_command,
        "artifact_predictions_sha256": sha256_file(artifact / "predictions.npz"),
        "metrics_sha256": sha256_file(scored / "metrics.json"),
    }


def _execute(job: dict, run_dir: Path, log_path: Path) -> dict:
    if job["phase"] == "P1":
        if job["model"] == "cytobridge":
            return _screen_cytobridge(job, run_dir, log_path)
        if job["model"] in {"chemcpa", "biolord"}:
            return _screen_external(job, run_dir, log_path)
        if job["model"] == "ridge":
            return _screen_ridge(run_dir, log_path)
        raise ValueError(f"Unsupported P1 model: {job['model']}")

    paths = _dataset_paths(job["dataset"])
    if job["model"].startswith("cytobridge"):
        result = _run_final_cytobridge(job, run_dir, log_path, paths)
    elif job["model"] in {"random", "mean", "oracle"}:
        result = _run_reference(job, run_dir, log_path, paths)
    elif job["model"] == "ridge":
        result = _run_final_ridge(job, run_dir, log_path, paths)
    elif job["model"] in {"chemcpa", "biolord"}:
        result = _run_final_external(job, run_dir, log_path, paths)
    else:
        raise ValueError(f"Unsupported final campaign model: {job['model']}")
    return {**result, **_package_and_score(job, run_dir, log_path, paths, result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    manifest, job = _load_job(args.manifest.resolve(), args.job_id)
    gate_hashes = _require_dependencies(job)
    gate_hashes.update(_require_learned_launch_authorization(job))
    gate_hashes.update(_require_source_tree())
    run_dir = RUN_ROOT / job["job_id"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse campaign run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "run.log"
    provenance = {
        "schema_version": 1,
        "job": job,
        "manifest_sha256": sha256_file(args.manifest),
        "campaign_spec_sha256": manifest.get("spec_sha256"),
        "gate_hashes": gate_hashes,
        "automatic_retry": False,
    }
    _write_json(run_dir / "provenance.json", provenance)
    try:
        details = _execute(job, run_dir, log_path)
        status = {
            "schema_version": 1,
            "status": "complete",
            "job_id": job["job_id"],
            "details": details,
            "run_log_sha256": sha256_file(log_path),
            "automatic_retry": False,
        }
        _write_json(run_dir / "run.status", status)
        print(json.dumps(status, sort_keys=True, default=str))
    except BaseException as exc:
        status = {
            "schema_version": 1,
            "status": "failed",
            "job_id": job["job_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            "automatic_retry": False,
        }
        _write_json(run_dir / "run.status", status)
        raise


if __name__ == "__main__":
    main()
