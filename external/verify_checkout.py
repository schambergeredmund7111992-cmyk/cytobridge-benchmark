"""Verify clean official external-baseline checkouts against pinned commits."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

def validate_checkout_state(
    *,
    label: str,
    expected_commit: str,
    observed_commit: str,
    porcelain_status: str,
) -> dict:
    if observed_commit != expected_commit:
        raise ValueError(
            f"{label} commit mismatch: expected {expected_commit}, observed {observed_commit}."
        )
    if porcelain_status.strip():
        raise ValueError(
            f"{label} checkout has uncommitted changes and is not admissible."
        )
    return {"label": label, "commit": observed_commit, "clean": True}


def verify_checkout(label: str, path: Path, expected_commit: str) -> dict:
    observed = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    )
    return validate_checkout_state(
        label=label,
        expected_commit=expected_commit,
        observed_commit=observed,
        porcelain_status=status,
    )


def validate_chemcpa_policy(policy: dict) -> dict:
    if policy.get("initialization") != "from_scratch":
        raise ValueError("chemCPA source policy must require from-scratch initialization.")
    if policy.get("pretrained_weights_allowed") is not False:
        raise ValueError("chemCPA source policy must forbid pretrained weights.")
    return {
        "chemcpa_initialization": "from_scratch",
        "pretrained_weights_used": False,
    }


def write_or_verify_identical(path: Path, payload: dict) -> bool:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(
                f"Existing checkout verification differs from current state: {path}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--chemcpa", type=Path, required=True)
    parser.add_argument("--biolord", type=Path, required=True)
    parser.add_argument("--biolord-repro", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(args.source_manifest.read_text())
    results = [
        verify_checkout("chemcpa", args.chemcpa, pins["chemCPA"]["official_commit"]),
        verify_checkout(
            "biolord", args.biolord, pins["biolord"]["official_package_commit"]
        ),
        verify_checkout(
            "biolord_repro",
            args.biolord_repro,
            pins["biolord"]["reproducibility_commit"],
        ),
    ]
    chemcpa_policy = validate_chemcpa_policy(pins["chemCPA"])
    payload = {
        "schema_version": 1,
        "checkouts": results,
        **chemcpa_policy,
    }
    created = write_or_verify_identical(args.out, payload)
    print(json.dumps({"verified": len(results), "created": created}))


if __name__ == "__main__":
    main()
