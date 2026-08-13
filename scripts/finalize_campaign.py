"""Aggregate all completed P2/P3 jobs and unlock paper inputs through the P4 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.aggregate_benchmark import aggregate_benchmark
from eval.artifacts import sha256_file, validate_artifact
from release.collect_gradient_audits import collect_gradient_audits
from release.generate_paper_inputs import generate_paper_inputs
from release.result_gate import build_result_manifest

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
REQUIREMENTS = PROJECT_ROOT / "release/RESULT_REQUIREMENTS.json"

PRIMARY_MAIN_MODELS = {
    "cytobridge_full",
    "chemcpa",
    "biolord",
    "random",
    "mean",
    "ridge",
    "oracle",
}
ABLATION_MODELS = {
    "cytobridge_full",
    "cytobridge_reconstruction_free",
    "cytobridge_normalized_reconstruction",
    "cytobridge_no_pathway_gate",
    "cytobridge_drug_blind",
}


def _complete_job(job: dict) -> tuple[Path, dict]:
    run_dir = RUN_ROOT / job["job_id"]
    status_path = run_dir / "run.status"
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        raise ValueError(f"Campaign job is not complete: {job['job_id']}")
    return run_dir, status


def _artifact_row(job: dict) -> dict:
    run_dir, _ = _complete_job(job)
    artifact_dir = run_dir / "artifact"
    scored_dir = run_dir / "scored"
    artifact = validate_artifact(artifact_dir)
    return {
        "model": str(artifact.provenance["model"]),
        "seed": int(artifact.provenance["seed"]),
        "artifact_dir": str(artifact_dir),
        "scored_dir": str(scored_dir),
        "job_id": job["job_id"],
    }


def _write_run_manifest(path: Path, jobs: list[dict]) -> None:
    rows = [_artifact_row(job) for job in jobs]
    table = pd.DataFrame(rows)
    if table[["model", "seed"]].astype(str).duplicated().any():
        raise ValueError(f"Aggregate run manifest has duplicate model/seed rows: {path}")
    table.to_csv(path, index=False)


def finalize_campaign(manifest_path: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite P4 release output: {output_root}")
    campaign = json.loads(manifest_path.read_text())
    jobs = campaign["jobs"]
    for job in jobs:
        _complete_job(job)
    output_root.mkdir(parents=True)
    manifests = output_root / "run_manifests"
    manifests.mkdir()

    primary_jobs = [
        job
        for job in jobs
        if job["phase"] == "P2"
        and job["dataset"] == "sciplex_drug_disjoint_v2"
        and job["model"] in PRIMARY_MAIN_MODELS
    ]
    scaffold_jobs = [
        job
        for job in jobs
        if job["phase"] == "P2"
        and job["dataset"] == "sciplex_scaffold_disjoint_v2"
    ]
    tahoe_jobs = [job for job in jobs if job["phase"] == "P3"]
    ablation_jobs = [
        job
        for job in jobs
        if job["phase"] == "P2"
        and job["dataset"] == "sciplex_drug_disjoint_v2"
        and job["model"] in ABLATION_MODELS
    ]
    grouped = {
        "primary": primary_jobs,
        "scaffold": scaffold_jobs,
        "tahoe": tahoe_jobs,
        "ablations": ablation_jobs,
    }
    aggregates = {}
    for name, selected_jobs in grouped.items():
        manifest = manifests / f"{name}.csv"
        _write_run_manifest(manifest, selected_jobs)
        aggregate = output_root / f"aggregate_{name}"
        aggregate_benchmark(
            manifest,
            aggregate,
            reference_model="cytobridge",
            n_boot=10_000,
            bootstrap_seed=7301,
        )
        aggregates[name] = aggregate

    gradient_jobs = [
        job
        for job in jobs
        if job["phase"] == "P2"
        and job["dataset"] == "sciplex_drug_disjoint_v2"
        and job["model"] in ABLATION_MODELS
    ]
    gradient_manifest = manifests / "gradient_audits.csv"
    pd.DataFrame(
        [
            {
                "model": job["model"],
                "seed": int(job["seed"]),
                "gradient_audit_path": str(
                    RUN_ROOT / job["job_id"] / "training/gradient_norms.jsonl"
                ),
            }
            for job in gradient_jobs
        ]
    ).to_csv(gradient_manifest, index=False)
    gradient_table = output_root / "gradient_audits.csv"
    collect_gradient_audits(gradient_manifest, gradient_table)

    result_manifest = output_root / "result_manifest.json"
    result = build_result_manifest(
        REQUIREMENTS,
        aggregates["primary"],
        aggregates["scaffold"],
        aggregates["tahoe"],
        aggregates["ablations"],
        gradient_table,
        result_manifest,
    )
    paper_inputs = output_root / "paper_inputs"
    generated = generate_paper_inputs(
        result_manifest,
        {
            "sciplex_drug_disjoint_v2": aggregates["primary"],
            "sciplex_scaffold_disjoint_v2": aggregates["scaffold"],
            "tahoe": aggregates["tahoe"],
            "ablations": aggregates["ablations"],
        },
        paper_inputs,
    )
    summary = {
        "schema_version": 1,
        "status": result["status"],
        "campaign_manifest_sha256": sha256_file(manifest_path),
        "requirements_sha256": sha256_file(REQUIREMENTS),
        "result_manifest_sha256": sha256_file(result_manifest),
        "paper_input_manifest_sha256": sha256_file(
            paper_inputs / "generated_manifest.json"
        ),
        "generated": generated,
    }
    (output_root / "p4.status.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENT_ROOT / "release",
    )
    args = parser.parse_args()
    result = finalize_campaign(args.manifest.resolve(), args.out.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
