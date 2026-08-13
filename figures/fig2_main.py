"""
figures/fig2_main.py
--------------------
Fig 2: main results on sci-Plex internal val.

Bars + 95% bootstrap CI for 5 baselines + CytoBridge:
    Mean / Ridge / scGPT-zs / GEARS / LLM-only / CytoBridge
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


# Method order (left → right) and result CSV paths.
METHODS = [
    ("Mean",       "results/mean_baseline.csv",       "#9C9C9C"),
    ("Ridge",      "results/ridge_baseline.csv",      "#4C78A8"),
    ("scGPT-zs",   "results/scgpt_zeroshot.csv",      "#72B7B2"),
    ("GEARS",      "results/gears.csv",               "#F58518"),
    ("LLM-only",   "results/llm_only.csv",            "#B279A2"),
    ("CytoBridge", "results/cytobridge_internal.csv", "#54A24B"),
]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    bs = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main() -> None:
    rows = []
    for method, path, _ in METHODS:
        p = Path(path)
        if not p.exists():
            print(f"[fig2] missing {p} — skipping {method}")
            continue
        df = pd.read_csv(p)
        if "spearman_top50" not in df.columns:
            print(f"[fig2] {p} has no spearman_top50 column — skipping {method}")
            continue
        vals = df["spearman_top50"].dropna().to_numpy()
        m, lo, hi = bootstrap_mean_ci(vals)
        rows.append({"method": method, "mean": m, "lo": lo, "hi": hi})
    if not rows:
        print("No result files found for Fig 2.")
        return

    out_dir = Path("figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    palette = {m: c for m, _, c in METHODS}
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(df["method"], df["mean"],
           color=[palette[m] for m in df["method"]],
           edgecolor="#333", linewidth=0.5)
    err_lower = (df["mean"] - df["lo"]).clip(lower=0)
    err_upper = (df["hi"] - df["mean"]).clip(lower=0)
    ax.errorbar(df["method"], df["mean"],
                yerr=[err_lower, err_upper],
                fmt="none", ecolor="#222", capsize=3, linewidth=0.8)
    # Annotate Δ vs ridge (the existential bar)
    if "Ridge" in df["method"].values and "CytoBridge" in df["method"].values:
        ridge_m = df.loc[df["method"] == "Ridge", "mean"].iloc[0]
        cyto_m = df.loc[df["method"] == "CytoBridge", "mean"].iloc[0]
        delta = cyto_m - ridge_m
        ax.set_title(f"Δ Spearman vs Ridge: {delta:+.3f}", fontsize=9)
    ax.set_ylabel("Spearman@50 (mean ± 95% CI)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_main.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_main.png", dpi=200, bbox_inches="tight")
    print(f"saved -> {out_dir}/fig2_main.{{pdf,png}}")


if __name__ == "__main__":
    main()
