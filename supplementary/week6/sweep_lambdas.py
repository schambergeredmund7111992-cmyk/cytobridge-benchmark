#!/usr/bin/env python
"""T2-2: Lambda Sweep — grid search over lam_delta × lam_contrast.
===================================================================
Grid-search across:
  lam_delta    ∈ {0.0, 0.1, 0.5, 1.0}
  lam_contrast ∈ {0.1, 0.5, 1.0, 2.0}

Records val/spearman_top50 (best per combination from checkpoint filenames).

Usage:
  # Dry-run (print the command)
  python scripts/sweep_lambdas.py --dry-run

  # Run the full 4×4 sweep via Hydra multirun
  python scripts/sweep_lambdas.py

  # Parse results from completed runs (no re-run)
  python scripts/sweep_lambdas.py --parse-only
"""
from __future__ import annotations

import argparse, os, re, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
LAM_DELTA_VALUES   = [0.0, 0.1, 0.5, 1.0]
LAM_CONTRAST_VALUES = [0.1, 0.5, 1.0, 2.0]
CKPT_ROOT = Path("ckpts/t2_lambda_sweep")


def ckpt_dir_for(ld: float, lc: float) -> Path:
    """Path where checkpoints for a given (lam_delta, lam_contrast) are stored."""
    return CKPT_ROOT / f"ld_{ld}_lc_{lc}"


def extract_best_spearman(ckpt_dir: Path) -> float | None:
    """Parse checkpoint filenames (epoch*val_spearman*.ckpt) for the best Spearman."""
    best = None
    pattern = re.compile(r"val_spearman(\d+\.\d+)")
    for ckpt in ckpt_dir.glob("epoch*val_spearman*.ckpt"):
        m = pattern.search(ckpt.name)
        if m:
            val = float(m.group(1))
            if best is None or val > best:
                best = val
    return best


def build_cmd() -> str:
    """Construct the Hydra multirun command with correct env.

    Uses root v1 config + explicit overrides. Each run's ckpt.dirpath is
    unique via Hydra interpolation on the per-job loss.lam_delta/lam_contrast.
    """
    ld_str = ",".join(str(v) for v in LAM_DELTA_VALUES)
    lc_str = ",".join(str(v) for v in LAM_CONTRAST_VALUES)
    python = os.environ.get("PYTHON", "python")
    ld_preload = os.environ.get("LD_PRELOAD", "")
    pythonpath = os.environ.get("PYTHONPATH", "")
    preamble = ""
    if ld_preload:
        preamble += f"LD_PRELOAD={ld_preload} "
    if pythonpath:
        preamble += f"PYTHONPATH={pythonpath} "
    # \${...} passes ${...} literally through bash to Hydra, which resolves
    # the interpolation per job to the correct lam_delta/lam_contrast value.
    return (
        f"{preamble}{python} train.py -m "
        f"model.residual_decoder=true "
        f"model.pool_mode=drug_query "
        f"loss.lam_direction=0.0 "
        f"loss.lam_delta={ld_str} "
        f"loss.lam_contrast={lc_str} "
        f"run_name=t2_lambda_sweep "
        f"wandb.use=false "
        f"trainer.max_epochs=1 "
        f"trainer.precision=32 "
        f"ckpt.dirpath=ckpts/t2_lambda_sweep/ld_\\${{loss.lam_delta}}_lc_\\${{loss.lam_contrast}}"
    )


def parse_results() -> pd.DataFrame:
    """Scan ckpt directories and extract val/spearman_top50 per combination."""
    rows = []
    for ld in LAM_DELTA_VALUES:
        for lc in LAM_CONTRAST_VALUES:
            d = ckpt_dir_for(ld, lc)
            best = extract_best_spearman(d) if d.exists() else None
            rows.append({
                "lam_delta": ld,
                "lam_contrast": lc,
                "val_spearman_top50": best if best is not None else float("nan"),
            })
    return pd.DataFrame(rows)


def print_grid(df: pd.DataFrame):
    """Pretty-print pivot table + best combo."""
    pivot = df.pivot_table(
        index="lam_delta", columns="lam_contrast",
        values="val_spearman_top50", aggfunc="first",
    )
    pivot = pivot.reindex(columns=LAM_CONTRAST_VALUES, index=LAM_DELTA_VALUES)

    print("\n" + "=" * 72)
    print("  T2-2  LAMBDA SWEEP  —  val/spearman_top50")
    print("=" * 72)
    print("Rows: lam_delta  |  Cols: lam_contrast")
    print()
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}" if not np.isnan(x) else "   N/A"))

    valid = df.dropna(subset=["val_spearman_top50"])
    if len(valid) > 0:
        best = valid.loc[valid["val_spearman_top50"].idxmax()]
        print(f"\nBest: lam_delta={best['lam_delta']}, "
              f"lam_contrast={best['lam_contrast']}, "
              f"val_spearman_top50={best['val_spearman_top50']:.4f}")
    return pivot


def main():
    ap = argparse.ArgumentParser(description="T2-2 Lambda Sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--out", default="results/t2_lambda_sweep.csv")
    args = ap.parse_args()

    n = len(LAM_DELTA_VALUES) * len(LAM_CONTRAST_VALUES)

    if args.dry_run:
        print(f"Grid: {len(LAM_DELTA_VALUES)} lam_delta × "
              f"{len(LAM_CONTRAST_VALUES)} lam_contrast = {n} runs")
        print(f"lam_delta:   {LAM_DELTA_VALUES}")
        print(f"lam_contrast: {LAM_CONTRAST_VALUES}")
        print(f"\n{build_cmd()}")
        return

    if args.parse_only:
        df = parse_results()
        if df["val_spearman_top50"].isna().all():
            print("No checkpoint results found. Run the sweep first (omit --parse-only).")
            return
        print_grid(df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
        return

    # ---- Run ----
    print(f"T2-2 Lambda Sweep: {n} combinations")
    cmd = build_cmd()
    print(f"\n{cmd}\n")
    sys.stdout.flush()

    rc = subprocess.run(cmd, shell=True, cwd=os.getcwd()).returncode
    if rc != 0:
        print(f"\nSweep exited with code {rc}; parsing partial results...")

    df = parse_results()
    if df["val_spearman_top50"].isna().all():
        print("ERROR: No results found.")
        sys.exit(1)

    print_grid(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
