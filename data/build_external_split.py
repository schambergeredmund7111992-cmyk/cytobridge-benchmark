"""
data/build_external_split.py
----------------------------
Build Tahoe-100M external splits (2-tier) after sci-Plex preprocessing.

Splits:
  external_1: unseen drug, seen cell line
  external_2: seen drug, unseen cell line

The third tier (both unseen) was dropped because Tahoe-100M slices yield
too few samples for confident bootstrap CIs in that intersection.

Each split writes:
  * `tahoe_<split>.h5ad`     — sliced AnnData containing the union of the
                                sampled treated cells AND their paired
                                same-cell-line vehicle controls.
  * `tahoe_<split>.parquet`  — manifest with split-local positions:
        cell_idx          : row in the sliced h5ad of the treated cell
        control_cell_idx  : row in the sliced h5ad of the matched control
        original_idx      : original row in the full Tahoe AnnData
        original_control_idx : original row of the control cell
        drug_id, cell_line

The manifest is what `cytobridge.data.CytoBridgeDataset` consumes; treating
split-local positions as if they were original Tahoe indices would either
overflow the sliced arrays or silently misalign every sample, so we always
emit split-local positions here and keep originals in separate columns.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scanpy as sc
except Exception:  # pragma: no cover - CLI help must survive broken optional stacks too
    sc = None


DEFAULT_CONTROL_KEYWORDS = ("dmso", "vehicle", "control", "ctrl", "untreated")


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def _is_control_drug(series: pd.Series, keywords: tuple[str, ...]) -> pd.Series:
    pattern = "|".join(re.escape(k) for k in keywords)
    return series.astype(str).str.lower().str.contains(pattern, regex=True, na=False)


def _pair_with_controls(
    treated: pd.DataFrame,
    control_pool: pd.DataFrame,
    cell_col: str,
    seed: int,
) -> pd.DataFrame:
    """Attach a per-row matched control_idx by sampling within the same cell line.
    Returns the input frame filtered to rows for which a control was found,
    plus a `control_idx` column referring to original Tahoe row positions.
    """
    rng = np.random.default_rng(seed)
    by_cell = {
        cl: idx.to_numpy()
        for cl, idx in control_pool.groupby(cell_col).groups.items()
    }
    chosen = []
    skipped_cells: dict[str, int] = {}
    for _, row in treated.iterrows():
        pool = by_cell.get(row[cell_col])
        if pool is None or len(pool) == 0:
            skipped_cells[row[cell_col]] = skipped_cells.get(row[cell_col], 0) + 1
            chosen.append(-1)
            continue
        chosen.append(int(rng.choice(pool)))
    treated = treated.copy()
    treated["control_idx"] = chosen
    if skipped_cells:
        msg = ", ".join(f"{cl}: {n}" for cl, n in sorted(skipped_cells.items())[:5])
        print(f"[tahoe]   skipped {sum(skipped_cells.values())} treated cells "
              f"with no matched control (per cell-line: {msg}{'...' if len(skipped_cells) > 5 else ''})")
    return treated[treated["control_idx"] >= 0].copy()


def build_splits(
    tahoe_h5ad: Path,
    sciplex_splits_json: Path,
    sciplex_manifest: Path,
    out_dir: Path,
    drug_col: str = "drug",
    cell_col: str = "cell_line",
    n_per_split: int = 50000,
    seed: int = 42,
    control_keywords: tuple[str, ...] = DEFAULT_CONTROL_KEYWORDS,
) -> None:
    if sc is None:
        raise ImportError(
            "scanpy is required to build Tahoe external splits. Install the "
            "project environment before running this command."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(tahoe_h5ad)
    split_meta = json.load(open(sciplex_splits_json))
    sciplex_drugs = set(split_meta["train_drugs"] + split_meta["val_drugs"] + split_meta["test_drugs"])
    sciplex_cells = set(pd.read_parquet(sciplex_manifest)[cell_col].unique())

    obs = adata.obs[[drug_col, cell_col]].copy()
    obs["_idx"] = np.arange(adata.n_obs)
    is_control = _is_control_drug(obs[drug_col], control_keywords)
    treated_mask = ~is_control
    print(f"[tahoe] {is_control.sum()} control cells, {treated_mask.sum()} treated cells "
          f"(matched on keywords {list(control_keywords)})")

    masks = {
        "external_1": treated_mask
            & (~obs[drug_col].isin(sciplex_drugs))
            & (obs[cell_col].isin(sciplex_cells)),
        "external_2": treated_mask
            & (obs[drug_col].isin(sciplex_drugs))
            & (~obs[cell_col].isin(sciplex_cells)),
    }
    for offset, (name, mask) in enumerate(masks.items()):
        sub_treated = _sample_rows(obs[mask], n_per_split, seed + offset)
        if len(sub_treated) == 0:
            print(f"[tahoe] {name}: 0 cells after filtering — skipping write.")
            continue

        # Pair with matched same-cell-line controls before the slice so we know
        # which original rows must be packed into the sliced AnnData.
        cell_lines_in_split = set(sub_treated[cell_col].unique())
        local_controls = obs[is_control & obs[cell_col].isin(cell_lines_in_split)]
        if local_controls.empty:
            print(f"[tahoe] {name}: no control rows for any of the {len(cell_lines_in_split)} "
                  "split cell-lines — skipping write. Re-run with --control_keywords if your "
                  "Tahoe slice uses a different vehicle label.")
            continue
        paired = _pair_with_controls(
            treated=sub_treated, control_pool=local_controls,
            cell_col=cell_col, seed=seed + 10_000 + offset,
        )
        if paired.empty:
            print(f"[tahoe] {name}: every sampled treated cell lacked a same-cell-line "
                  "control — skipping write.")
            continue

        # Build split-local indices.
        treated_originals = paired["_idx"].to_numpy()
        control_originals = paired["control_idx"].to_numpy()
        union = np.unique(np.concatenate([treated_originals, control_originals]))
        original_to_local = {int(orig): i for i, orig in enumerate(union)}
        local_treated = np.fromiter(
            (original_to_local[int(i)] for i in treated_originals),
            dtype=np.int64, count=len(treated_originals),
        )
        local_control = np.fromiter(
            (original_to_local[int(i)] for i in control_originals),
            dtype=np.int64, count=len(control_originals),
        )

        manifest = pd.DataFrame({
            "cell_idx": local_treated,
            "control_cell_idx": local_control,
            "original_idx": treated_originals,
            "original_control_idx": control_originals,
            "drug_id": paired[drug_col].to_numpy(),
            "cell_line": paired[cell_col].to_numpy(),
        })
        manifest.to_parquet(out_dir / f"tahoe_{name}.parquet")
        adata[union].write_h5ad(out_dir / f"tahoe_{name}.h5ad")
        print(f"[tahoe] {name}: {len(manifest)} treated cells "
              f"({len(union)} rows in sliced h5ad incl. controls) -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tahoe_h5ad", type=Path, required=True)
    parser.add_argument("--sciplex_splits_json", type=Path,
                        default=Path("data/processed/sciplex_accept/drug_disjoint_v2/split_assignments.csv"))
    parser.add_argument("--sciplex_manifest", type=Path,
                        default=Path("data/processed/sciplex_accept/drug_disjoint_v2/splits/sciplex_train.parquet"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed/tahoe/splits"))
    parser.add_argument("--n_per_split", type=int, default=50000)
    parser.add_argument("--drug_col", default="drug")
    parser.add_argument("--cell_col", default="cell_line")
    parser.add_argument("--control_keywords", nargs="+",
                        default=list(DEFAULT_CONTROL_KEYWORDS),
                        help="Substrings (case-insensitive) used to identify vehicle/control "
                             "rows in the Tahoe drug column.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise SystemExit(
        "LEGACY TAHOE ENTRYPOINT DISABLED: external_1/external_2 is not part of "
        "protocol 1.4. Build data/processed/tahoe_accept with "
        "python -m data.preprocess_tahoe instead."
    )
    build_splits(
        tahoe_h5ad=args.tahoe_h5ad,
        sciplex_splits_json=args.sciplex_splits_json,
        sciplex_manifest=args.sciplex_manifest,
        out_dir=args.out_dir,
        drug_col=args.drug_col,
        cell_col=args.cell_col,
        n_per_split=args.n_per_split,
        seed=args.seed,
        control_keywords=tuple(args.control_keywords),
    )


if __name__ == "__main__":
    main()
