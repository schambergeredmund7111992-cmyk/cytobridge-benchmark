"""Generate LaTeX tables only from a validated, unlocked result manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.artifacts import sha256_file

RESULT_KEYS = {
    "sciplex_drug_disjoint_v2": "primary",
    "sciplex_scaffold_disjoint_v2": "scaffold",
    "tahoe": "tahoe",
    "ablations": "ablations",
}


def _latex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _estimate_ci(row: pd.Series, metric: str) -> str:
    if metric == "conditional":
        estimate = row["conditional_accuracy"]
        low = row["conditional_ci_low"]
        high = row["conditional_ci_high"]
    else:
        estimate = row["pair_own_spearman_top50"]
        low = row["spearman_ci_low"]
        high = row["spearman_ci_high"]
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def _table_tex(table: pd.DataFrame, caption: str, label: str) -> str:
    required = {
        "model",
        "conditional_accuracy",
        "conditional_ci_low",
        "conditional_ci_high",
        "pair_own_spearman_top50",
        "spearman_ci_low",
        "spearman_ci_high",
        "n_seeds",
    }
    if missing := required - set(table.columns):
        raise ValueError(f"Leaderboard is missing paper columns: {sorted(missing)}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Conditional accuracy (95\% CI) & Spearman@50 (95\% CI) & Seeds \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        series = pd.Series(row._asdict())
        lines.append(
            f"{_latex_escape(series['model'])} & {_estimate_ci(series, 'conditional')} & "
            f"{_estimate_ci(series, 'spearman')} & {int(series['n_seeds'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def generate_paper_inputs(
    result_manifest_path: Path,
    aggregate_dirs: dict[str, Path],
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite generated paper inputs: {output_dir}"
        )
    result_manifest = json.loads(result_manifest_path.read_text())
    if result_manifest.get("status") != "paper_inputs_unlocked":
        raise ValueError("Result manifest has not unlocked paper inputs.")
    if set(aggregate_dirs) != set(RESULT_KEYS):
        raise ValueError(
            "Aggregate directory mapping differs from the required result sets."
        )
    output_dir.mkdir(parents=True)
    generated_hashes = {}
    captions = {
        "sciplex_drug_disjoint_v2": "Primary sci-Plex drug-disjoint benchmark.",
        "sciplex_scaffold_disjoint_v2": "sci-Plex scaffold-disjoint stress test.",
        "tahoe": "Tahoe-100M balanced-panel benchmark.",
        "ablations": "Registered CytoBridge ablations on sci-Plex.",
    }
    for result_key, short_name in RESULT_KEYS.items():
        leaderboard_path = aggregate_dirs[result_key] / "leaderboard.csv"
        expected_hash = result_manifest["results"][result_key]["leaderboard_sha256"]
        if sha256_file(leaderboard_path) != expected_hash:
            raise ValueError(f"Leaderboard hash drift for {result_key}.")
        output_path = output_dir / f"table_{short_name}.tex"
        output_path.write_text(
            _table_tex(
                pd.read_csv(leaderboard_path),
                captions[result_key],
                f"tab:{short_name}",
            )
        )
        generated_hashes[output_path.name] = sha256_file(output_path)
    manifest_hash = sha256_file(result_manifest_path)
    ready_path = output_dir / "results_ready.tex"
    ready_path.write_text(
        "% Generated only after release/result_gate.py passed.\n"
        f"\\newcommand{{\\ResultsManifestHash}}{{{manifest_hash}}}\n"
        + "\n".join(
            f"\\input{{generated/table_{short_name}.tex}}"
            for short_name in RESULT_KEYS.values()
        )
        + "\n"
    )
    generated_hashes[ready_path.name] = sha256_file(ready_path)
    output = {
        "result_manifest_sha256": manifest_hash,
        "generated_hashes": generated_hashes,
        "scientific_prose_generated": False,
    }
    (output_dir / "generated_manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--tahoe", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate_paper_inputs(
        args.result_manifest,
        {
            "sciplex_drug_disjoint_v2": args.primary,
            "sciplex_scaffold_disjoint_v2": args.scaffold,
            "tahoe": args.tahoe,
            "ablations": args.ablations,
        },
        args.out,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
