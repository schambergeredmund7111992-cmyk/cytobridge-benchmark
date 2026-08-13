"""Collect first-batch per-loss gradient audits from a frozen run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_gradient_audits(manifest_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite gradient audit table: {output_path}"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"model", "seed", "gradient_audit_path"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Gradient run manifest is missing columns: {sorted(missing)}")
    if manifest[["model", "seed"]].astype(str).duplicated().any():
        raise ValueError("Gradient run manifest model/seed rows must be unique.")
    records = []
    for run in manifest.itertuples(index=False):
        path = Path(str(run.gradient_audit_path))
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Expected exactly one gradient audit record in {path}.")
        payload = json.loads(lines[0])
        components = payload.get("components")
        if not isinstance(components, dict) or not components:
            raise ValueError(f"Gradient audit {path} contains no active components.")
        for component, values in sorted(components.items()):
            norm = float(values["weighted_gradient_l2"])
            connected = int(values["connected_parameter_tensors"])
            if not np.isfinite(norm) or connected <= 0:
                raise ValueError(
                    f"Inactive or non-finite audited gradient for {run.model}/{run.seed}/"
                    f"{component}."
                )
            records.append(
                {
                    "model": str(run.model),
                    "seed": int(run.seed),
                    "component": str(component),
                    "coefficient": float(values["coefficient"]),
                    "raw_loss": float(values["raw_loss"]),
                    "weighted_gradient_l2": norm,
                    "connected_parameter_tensors": connected,
                    "source_path": str(path),
                }
            )
    output = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = collect_gradient_audits(args.manifest, args.out)
    print(
        json.dumps(
            {
                "rows": len(output),
                "runs": output[["model", "seed"]].drop_duplicates().shape[0],
            }
        )
    )


if __name__ == "__main__":
    main()
