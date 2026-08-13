"""Write row counts and storage measurements used for the launch estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import pandas as pd


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite preparation summary: {args.out}")
    raw = args.data_root / "raw"
    processed = args.data_root / "processed" / "sciplex_accept"
    official = raw / "sciplex" / "SrivatsanTrapnell2020_sciplex3.h5ad"
    hallmark = raw / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt"
    report = {
        "schema_version": 1,
        "official_sciplex": {
            "bytes": official.stat().st_size,
            "md5": _hash(official, "md5"),
        },
        "hallmark": {
            "bytes": hallmark.stat().st_size,
            "sha256": _hash(hallmark, "sha256"),
        },
        "protocols": {},
    }
    for protocol in ("drug_disjoint_v2", "scaffold_disjoint_v2"):
        base = processed / protocol
        h5ad = ad.read_h5ad(base / "sciplex_processed.h5ad", backed="r")
        splits = {}
        for split in ("train", "val", "test"):
            table = pd.read_parquet(base / "splits" / f"sciplex_{split}.parquet")
            splits[split] = {
                "rows": len(table),
                "drugs": int(table["drug_id"].astype(str).nunique()),
                "contexts": int(table["context_id"].astype(str).nunique()),
            }
        report["protocols"][protocol] = {
            "processed_shape": [int(h5ad.n_obs), int(h5ad.n_vars)],
            "splits": splits,
            "directory_bytes": sum(
                path.stat().st_size for path in base.rglob("*") if path.is_file()
            ),
        }
        h5ad.file.close()
    usage = shutil.disk_usage(args.data_root)
    report["storage"] = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["protocols"], sort_keys=True))


if __name__ == "__main__":
    main()

