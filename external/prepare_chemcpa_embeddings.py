"""Create training-normalized official RDKit2D inputs for chemCPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external.chemcpa_runtime import build_rdkit_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_rdkit_embeddings(args.export, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
