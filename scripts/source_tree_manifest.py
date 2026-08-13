"""Create or verify a deterministic source-tree hash when no Git repository is present."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "ckpts",
    "logs",
    "experiments",
    "outputs",
    "wandb",
}
EXCLUDED_DATA_PARTS = {"raw", "processed", "cache", "smoke"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    project_root: Path, *, exclude_paths: tuple[Path, ...] = ()
) -> dict:
    excluded = {path.resolve() for path in exclude_paths}
    files = {}
    for path in sorted(project_root.rglob("*")):
        if (
            not path.is_file()
            or path.name == ".DS_Store"
            or path.resolve() in excluded
        ):
            continue
        relative = path.relative_to(project_root)
        if set(relative.parts) & EXCLUDED_PARTS or any(
            part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if len(relative.parts) >= 3 and relative.parts[:2] == ("code", "data"):
            if relative.parts[2] in EXCLUDED_DATA_PARTS:
                continue
        files[str(relative)] = _sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "algorithm": "sha256(path-to-sha256 canonical JSON)",
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    observed = build_manifest(
        args.project_root.resolve(), exclude_paths=(args.manifest,)
    )
    if args.command == "write":
        if args.manifest.exists():
            raise FileExistsError(f"Refusing to overwrite source manifest: {args.manifest}")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
    else:
        expected = json.loads(args.manifest.read_text())
        if expected != observed:
            raise ValueError(
                f"Source tree differs from manifest: {observed['tree_sha256']} != "
                f"{expected.get('tree_sha256')}"
            )
    print(json.dumps({"files": len(observed["files"]), "sha256": observed["tree_sha256"]}))


if __name__ == "__main__":
    main()
