from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default="perturblab/scgpt-human")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.out,
        allow_patterns=["args.json", "best_model.pt", "vocab.json", "README.md"],
    )
    print(path)


if __name__ == "__main__":
    main()
