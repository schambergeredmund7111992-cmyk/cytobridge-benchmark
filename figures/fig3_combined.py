"""
figures/fig3_combined.py
------------------------
Fig 3 (6 panels):
    a) External-1 unseen drug Spearman
    b) External-2 unseen cell line Spearman
    c) Replogle negative-control Spearman (should be ≈ 0)
    d) GDSC2 bulk transfer Spearman
    e) 4 ablation grid (no_pathway_gating / no_pathway_loss / no_contrast / no_molformer)
    f) Pathway attribution heatmap: predicted [n_drugs, K=50] vs GSEA truth

All panels degrade gracefully if a CSV is missing — the panel just shows
"data missing" rather than crashing the whole figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def _read_spearman(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([])
    df = pd.read_csv(path)
    return df.get("spearman_top50", pd.Series(dtype=float)).dropna().to_numpy()


def _missing(ax, label: str) -> None:
    ax.text(0.5, 0.5, f"missing\n{label}", ha="center", va="center",
            transform=ax.transAxes, color="#999", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_split_bar(ax, label: str, csv_path: Path, color: str) -> None:
    vals = _read_spearman(csv_path)
    if vals.size == 0:
        _missing(ax, csv_path.name)
        ax.set_title(label, fontsize=9)
        return
    rng = np.random.default_rng(0)
    boot = rng.choice(vals, size=(1000, len(vals)), replace=True).mean(axis=1)
    m, lo, hi = vals.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975)
    ax.bar([label], [m], color=color, edgecolor="#333", linewidth=0.4)
    ax.errorbar([label], [m], yerr=[[m - lo], [hi - m]],
                fmt="none", ecolor="#222", capsize=3, linewidth=0.8)
    ax.axhline(0, color="#666", linewidth=0.5)
    ax.set_ylabel("Spearman@50")
    ax.set_title(label, fontsize=9)
    ax.set_xticks([])


def _panel_ablation(ax) -> None:
    summary = Path("results/ablation_summary.csv")
    if summary.exists():
        df = pd.read_csv(summary)
    else:
        rows = []
        for ab in ["no_pathway_gating", "no_pathway_loss", "no_contrast", "no_molformer"]:
            p = Path(f"results/cytobridge_internal_{ab}.csv")
            vals = _read_spearman(p)
            rows.append({"ablation": ab, "spearman_top50": vals.mean() if vals.size else np.nan})
        v1 = _read_spearman(Path("results/cytobridge_internal.csv"))
        rows.insert(0, {"ablation": "v1 (full)",
                        "spearman_top50": v1.mean() if v1.size else np.nan})
        df = pd.DataFrame(rows).dropna(subset=["spearman_top50"])
    if df.empty:
        _missing(ax, "ablation results")
        ax.set_title("(e) Ablations", fontsize=9)
        return
    sns.barplot(data=df, x="ablation", y="spearman_top50", ax=ax,
                color="#54A24B", edgecolor="#333")
    ax.set_ylabel("Spearman@50")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("(e) Ablations", fontsize=9)


def _panel_pathway_heatmap(ax) -> None:
    pred_p = Path("results/pathway_attribution_pred.npy")
    true_p = Path("results/pathway_attribution_true.npy")
    if not (pred_p.exists() and true_p.exists()):
        _missing(ax, "pathway_attribution_*.npy")
        ax.set_title("(f) Pathway attribution", fontsize=9)
        return
    pred = np.load(pred_p)
    true = np.load(true_p)
    n = min(pred.shape[0], 30)         # top-30 drugs for readability
    pred_n = pred[:n] / np.maximum(pred[:n].sum(axis=1, keepdims=True), 1e-8)
    true_n = true[:n] / np.maximum(true[:n].sum(axis=1, keepdims=True), 1e-8)
    diff = pred_n - true_n
    im = ax.imshow(diff, cmap="coolwarm", vmin=-0.05, vmax=0.05, aspect="auto")
    ax.set_xlabel(f"Pathway (K={pred.shape[1]})")
    ax.set_ylabel(f"Drug (top {n})")
    ax.set_title("(f) Pathway attribution: pred − GSEA", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # Sanity-check correlation between predicted and true row-by-row
    rhos = []
    for i in range(n):
        r = spearmanr(pred[i], true[i]).statistic
        if not np.isnan(r):
            rhos.append(r)
    if rhos:
        ax.text(0.99, 0.01, f"med ρ={np.median(rhos):.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#222")


def main() -> None:
    out_dir = Path("figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.55, wspace=0.45)
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[0, 3]),
        "e": fig.add_subplot(gs[1, :2]),
        "f": fig.add_subplot(gs[1, 2:]),
    }
    _panel_split_bar(axes["a"], "(a) Ext-1 unseen drug",
                     Path("results/cytobridge_tahoe_external_1.csv"), "#F58518")
    _panel_split_bar(axes["b"], "(b) Ext-2 unseen cell",
                     Path("results/cytobridge_tahoe_external_2.csv"), "#E45756")
    _panel_split_bar(axes["c"], "(c) Replogle (neg ctrl)",
                     Path("results/cytobridge_replogle.csv"), "#9C9C9C")
    _panel_split_bar(axes["d"], "(d) GDSC2 bulk",
                     Path("results/gdsc2_transfer.csv"), "#4C78A8")
    _panel_ablation(axes["e"])
    _panel_pathway_heatmap(axes["f"])

    fig.suptitle("Fig 3 — Generalisation, ablation, interpretability", fontsize=10)
    fig.savefig(out_dir / "fig3_combined.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_combined.png", dpi=200, bbox_inches="tight")
    print(f"saved -> {out_dir}/fig3_combined.{{pdf,png}}")


if __name__ == "__main__":
    main()
