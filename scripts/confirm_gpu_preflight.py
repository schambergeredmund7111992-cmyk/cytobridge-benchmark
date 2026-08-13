"""Record explicit PI approval of an unchanged fresh GPU preflight draft."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.artifacts import sha256_file


def confirm(draft_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite confirmed preflight: {output_path}")
    draft = json.loads(draft_path.read_text())
    if (
        draft.get("status") != "draft"
        or draft.get("pi_confirmed") is not False
        or not draft.get("eligible_gpus")
    ):
        raise ValueError("GPU preflight draft is not eligible for confirmation.")
    payload = {
        **draft,
        "status": "passed",
        "pi_confirmed": True,
        "pi_confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        "draft_sha256": sha256_file(draft_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = confirm(args.draft, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
