"""Download only the pinned Tahoe metadata needed before bounded expression streaming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.artifacts import sha256_file

TAHOE_REVISION = "2dc5790"
METADATA_FILES = (
    "metadata/obs_metadata.parquet",
    "metadata/sample_metadata.parquet",
    "metadata/drug_metadata.parquet",
    "metadata/gene_metadata.parquet",
    "metadata/cell_line_metadata.parquet",
)


def download_tahoe_metadata(output_dir: Path, revision: str = TAHOE_REVISION) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix with an existing Tahoe metadata dir: {output_dir}"
        )
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="tahoebio/Tahoe-100M",
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(METADATA_FILES),
        local_dir=output_dir,
    )
    files = {name: output_dir / name for name in METADATA_FILES}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned Tahoe snapshot omitted metadata files: {missing}"
        )
    manifest = {
        "dataset": "tahoebio/Tahoe-100M",
        "revision": revision,
        "files": {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        },
        "expression_data_downloaded": False,
    }
    (output_dir / "metadata_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default=TAHOE_REVISION)
    args = parser.parse_args()
    result = download_tahoe_metadata(args.out, args.revision)
    print(json.dumps({"revision": result["revision"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
