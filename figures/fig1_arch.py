"""
figures/fig1_arch.py
--------------------
Figure 1: motivation panel (a) + architecture diagram (b) + pathway-gating illustration (c).

Run after main + ablation results are in results/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10})


def panel_a_motivation(ax, results_csv: str = "results/baseline_summary.csv"):
    """Show that ridge baseline is at parity with FMs (the motivating gap)."""
    if Path(results_csv).exists():
        df = pd.read_csv(results_csv)
        sns.barplot(data=df, x="method", y="spearman_mean", ax=ax)
        ax.set_ylabel("Per-pair Spearman ρ")
        ax.set_title("Linear baseline parity (Nat Methods 2025 gap)")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "Run baselines first.", ha="center", va="center",
                transform=ax.transAxes)


def panel_b_architecture(ax):
    """Architecture diagram drawn directly in matplotlib for reproducibility."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 4.2, 2.2, 0.8, "scRNA-seq\ncell state"),
        (0.3, 1.0, 2.2, 0.8, "SMILES\ncandidate drug"),
        (3.0, 4.2, 2.0, 0.8, "scGPT\nfrozen"),
        (3.0, 1.0, 2.0, 0.8, "MolFormer\nfrozen"),
        (5.8, 2.55, 2.1, 1.0, "Pathway-Gated\nCrossBridge"),
        (8.4, 3.6, 1.4, 0.7, "DEG\nprediction"),
        (8.4, 2.35, 1.4, 0.7, "Pathway\nattribution"),
        (8.4, 1.1, 1.4, 0.7, "CytoReasoner\nAPI agent"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#f7f7f7", edgecolor="#333333", lw=1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    arrows = [
        ((2.5, 4.6), (3.0, 4.6)),
        ((2.5, 1.4), (3.0, 1.4)),
        ((5.0, 4.6), (5.8, 3.2)),
        ((5.0, 1.4), (5.8, 2.9)),
        ((7.9, 3.25), (8.4, 3.95)),
        ((7.9, 3.0), (8.4, 2.7)),
        ((7.9, 2.75), (8.4, 1.45)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(6.85, 2.2, "MSigDB Hallmark\nK=50 prototypes", ha="center", va="center", fontsize=7)


def panel_c_pathway_gating(ax):
    """Toy illustration of pathway gating — heatmap over K=50 prototypes."""
    rng = np.random.default_rng(0)
    K = 50
    n = 6
    attn = rng.dirichlet(np.ones(K) * 0.3, size=n)
    # Sparsify to top-5
    attn_sparse = np.zeros_like(attn)
    for i in range(n):
        idx = np.argsort(-attn[i])[:5]
        attn_sparse[i, idx] = attn[i, idx]
    sns.heatmap(attn_sparse, ax=ax, cbar=True, cmap="viridis")
    ax.set_xlabel("MSigDB Hallmark prototype (K=50)")
    ax.set_ylabel("(drug, cell) pair")
    ax.set_title("Sparse pathway routing")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    panel_a_motivation(axes[0])
    panel_b_architecture(axes[1])
    panel_c_pathway_gating(axes[2])
    fig.tight_layout()
    out = Path("figures/out/fig1_arch.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
