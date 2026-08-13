"""Materialize and validate the frozen P1-P3 experiment campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _job(
    job_id: str,
    phase: str,
    dataset: str,
    model: str,
    *,
    seed: int | None,
    config_id: str,
    depends_on: list[str],
    learned: bool,
) -> dict:
    return {
        "job_id": job_id,
        "phase": phase,
        "dataset": dataset,
        "model": model,
        "config_id": config_id,
        "seed": seed,
        "depends_on": depends_on,
        "resources": {"gpus": 1 if learned else 0},
        "learned": learned,
        "command": [
            "python",
            "-m",
            "scripts.run_campaign_job",
            "--manifest",
            "../experiments/campaign_manifest.json",
            "--job-id",
            job_id,
        ],
        "expected_outputs": [
            f"experiments/runs/{job_id}/run.status",
            f"experiments/runs/{job_id}/provenance.json",
        ],
    }


def materialize(spec: dict) -> list[dict]:
    screening_seed = int(spec["screening_seed"])
    seeds = [int(seed) for seed in spec["final_seeds"]]
    jobs: list[dict] = []
    for index in range(1, 13):
        config_id = f"cytobridge_{index:02d}"
        jobs.append(
            _job(
                f"p1-sciplex-primary-{config_id}-seed{screening_seed}",
                "P1",
                "sciplex_drug_disjoint_v2",
                "cytobridge",
                seed=screening_seed,
                config_id=config_id,
                depends_on=["gate:p0_ready"],
                learned=True,
            )
        )
    for model, count in (("chemcpa", 6), ("biolord", 6)):
        for index in range(1, count + 1):
            config_id = f"{model}_{index:02d}"
            jobs.append(
                _job(
                    f"p1-sciplex-primary-{config_id}-seed{screening_seed}",
                    "P1",
                    "sciplex_drug_disjoint_v2",
                    model,
                    seed=screening_seed,
                    config_id=config_id,
                    depends_on=["gate:p0_ready"],
                    learned=True,
                )
            )
    jobs.append(
        _job(
            "p1-sciplex-primary-ridge-grid",
            "P1",
            "sciplex_drug_disjoint_v2",
            "ridge",
            seed=None,
            config_id="ridge_alpha_grid",
            depends_on=["gate:p0_ready"],
            learned=False,
        )
    )

    selected = ["gate:p1_selection_frozen"]
    primary_arms = spec["final"]["cytobridge_arms"]
    for arm in primary_arms:
        for seed in seeds:
            jobs.append(
                _job(
                    f"p2-sciplex-primary-{arm}-seed{seed}",
                    "P2",
                    "sciplex_drug_disjoint_v2",
                    arm,
                    seed=seed,
                    config_id="selected" if arm == "cytobridge_full" else arm,
                    depends_on=selected,
                    learned=True,
                )
            )
    for model in spec["final"]["external_models"]:
        for seed in seeds:
            jobs.append(
                _job(
                    f"p2-sciplex-primary-{model}-seed{seed}",
                    "P2",
                    "sciplex_drug_disjoint_v2",
                    model,
                    seed=seed,
                    config_id="selected",
                    depends_on=selected,
                    learned=True,
                )
            )
    for dataset, short in (
        ("sciplex_drug_disjoint_v2", "primary"),
        ("sciplex_scaffold_disjoint_v2", "scaffold"),
    ):
        if short == "scaffold":
            for model in ("cytobridge", *spec["final"]["external_models"]):
                for seed in seeds:
                    jobs.append(
                        _job(
                            f"p2-sciplex-{short}-{model}-seed{seed}",
                            "P2",
                            dataset,
                            model,
                            seed=seed,
                            config_id="selected_primary",
                            depends_on=selected,
                            learned=True,
                        )
                    )
        for model, model_seeds in (
            ("random", seeds),
            ("mean", [None]),
            ("ridge", [None]),
            ("oracle", [None]),
        ):
            for seed in model_seeds:
                suffix = f"seed{seed}" if seed is not None else "deterministic"
                jobs.append(
                    _job(
                        f"p2-sciplex-{short}-{model}-{suffix}",
                        "P2",
                        dataset,
                        model,
                        seed=seed,
                        config_id="selected_primary" if model == "ridge" else model,
                        depends_on=selected,
                        learned=False,
                    )
                )

    for seed in seeds:
        jobs.append(
            _job(
                f"p3-tahoe-cytobridge-seed{seed}",
                "P3",
                "tahoe",
                "cytobridge",
                seed=seed,
                config_id="selected_primary",
                depends_on=[*selected, "gate:tahoe_ready"],
                learned=True,
            )
        )
        jobs.append(
            _job(
                f"p3-tahoe-random-seed{seed}",
                "P3",
                "tahoe",
                "random",
                seed=seed,
                config_id="random",
                depends_on=["gate:tahoe_ready"],
                learned=False,
            )
        )
    for model in ("mean", "ridge", "oracle"):
        jobs.append(
            _job(
                f"p3-tahoe-{model}-deterministic",
                "P3",
                "tahoe",
                model,
                seed=None,
                config_id="tahoe_alpha_grid" if model == "ridge" else model,
                depends_on=["gate:tahoe_ready"],
                learned=False,
            )
        )
    return jobs


def validate(jobs: list[dict]) -> dict:
    ids = [job["job_id"] for job in jobs]
    outputs = [output for job in jobs for output in job["expected_outputs"]]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate job_id")
    if len(outputs) != len(set(outputs)):
        errors.append("duplicate expected output")
    learned = [job for job in jobs if job["learned"]]
    if len(learned) != 79:
        errors.append(f"expected 79 learned jobs, observed {len(learned)}")
    audits = [
        job
        for job in learned
        if job["dataset"] == "sciplex_drug_disjoint_v2"
        and job["phase"] == "P2"
        and job["model"].startswith("cytobridge")
    ]
    if len(audits) != 25:
        errors.append(f"expected 25 gradient-audit jobs, observed {len(audits)}")
    return {
        "ok": not errors,
        "errors": errors,
        "jobs": len(jobs),
        "learned_jobs": len(learned),
        "gradient_audit_jobs": len(audits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate"))
    parser.add_argument("--spec", type=Path, default=Path("../protocols/campaign.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        if args.manifest.exists():
            raise FileExistsError(f"Refusing to overwrite campaign manifest: {args.manifest}")
        spec = yaml.safe_load(args.spec.read_text())
        jobs = materialize(spec)
        validation = validate(jobs)
        if not validation["ok"]:
            raise ValueError(validation["errors"])
        payload = {
            "schema_version": 1,
            "protocol_version": spec["protocol_version"],
            "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            "jobs": jobs,
            "validation": validation,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        payload = json.loads(args.manifest.read_text())
        validation = validate(payload["jobs"])
        if not validation["ok"]:
            raise SystemExit(json.dumps(validation, sort_keys=True))
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
