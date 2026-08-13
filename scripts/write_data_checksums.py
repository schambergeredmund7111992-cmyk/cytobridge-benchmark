from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/checksums.json"))
    parser.add_argument("--patterns", nargs="+", default=["*.h5ad", "*.parquet", "*.npy", "*.npz", "*.csv", "*.gmt"])
    args = parser.parse_args()

    rows = []
    for pattern in args.patterns:
        for path in sorted(args.root.rglob(pattern)):
            if path.is_file():
                rows.append({
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} checksums -> {args.out}")


if __name__ == "__main__":
    main()
