"""Figure 4: Control validation (6 panels).

Reads the pooled-construction figures_data directory produced by
`scripts/regenerate_paper_numbers.py`; falls back to the legacy per-pair
artifacts with a warning.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figs"
os.makedirs(OUT, exist_ok=True)

TEAL, CORAL, SLATE, NEAR, GRID, GOLD = "#2A9D8F", "#E76F51", "#6C7A89", "#1F2937", "#E5E7EB", "#E9C46A"
DPI = 300
rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                 "axes.spines.right": False, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                 "axes.edgecolor": NEAR, "text.color": NEAR, "axes.labelcolor": NEAR,
                 "xtick.color": NEAR, "ytick.color": NEAR,
                 "xtick.labelsize": 7, "ytick.labelsize": 7})


def panel(ax, L, title=""):
    ax.set_title(f"{title}", fontsize=8.5, loc="left", pad=3)
    ax.text(-0.02, 1.16, L, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="figures_data directory (default: out/figures_data)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    data = args.data
    legacy = False
    if data is None:
        candidate = Path("out") / "figures_data"
        if candidate.is_dir():
            data = candidate
        else:
            legacy = True
            print("[fig4] WARNING: no pooled figures_data found; using legacy "
                  "per-pair artifacts (superseded construction).", file=sys.stderr)
            data = ROOT / "analysis" / "data"
            data2 = ROOT / "analysis" / "data2"
    data = Path(data)
    if legacy:
        data2 = Path(data2)

    print("Generating Figure 4: control validation...")
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)

    # (a) calibration curve
    ax = fig.add_subplot(gs[0, 0])
    cal = pd.read_csv(data / "calibration.csv")
    meta = json.load(open(data / "calibration_meta.json", encoding="utf-8"))
    best_auc = meta["cytobridge_auc"]
    ax.plot(cal.alpha, cal.auc, "-o", color=TEAL, ms=3, lw=1.4, zorder=3)
    ea = meta["effective_alpha"]
    ax.axhline(best_auc, color=CORAL, ls="--", lw=1)
    ax.axvline(ea, color=CORAL, ls="--", lw=1)
    ax.scatter([ea], [best_auc], color=CORAL, s=30, zorder=4)
    ax.text(ea + 0.04, 0.60, f"CytoBridge\nalpha={ea:.2f}", fontsize=6.5, color=CORAL)
    ax.set_xlabel("injected drug signal alpha")
    ax.set_ylabel("discrimination AUC")
    ax.set_ylim(0.45, 1.03)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "a", "control tracks real signal")

    # (b) ladder
    ax = fig.add_subplot(gs[0, 1])
    v = pd.read_csv(data / "ladder.csv").set_index("predictor")
    order_names = ["Random (50 perm)", "Mean (collapsed)", "Ridge (clean)",
                   "chemCPA (collapsed)", "biolord (collapsed)",
                   "CytoBridge (best)", "Oracle (truth)"]
    available = [n for n in order_names if n in v.index and not pd.isna(v.loc[n, "auc"])]
    vals = [v.loc[n, "auc"] for n in available]
    labels_short = ["Rand", "Mean", "Ridge", "chemCPA", "biolord", "CB", "Oracle"][: len(available)]
    colors_bar = [SLATE, SLATE, SLATE, SLATE, SLATE, CORAL, TEAL][: len(available)]
    ax.bar(range(len(available)), vals, color=colors_bar, width=0.66, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0.5, color=NEAR, ls=":", lw=0.8)
    ax.axhline(0.7, color=SLATE, ls="--", lw=0.8)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels_short, fontsize=6)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUC")
    for i, a in enumerate(vals):
        ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.5, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "b", "calibrated 0.5 to 1.0")

    # (c) per-config AUC
    ax = fig.add_subplot(gs[0, 2])
    if legacy:
        cm = pd.read_csv(data / "config_metrics.csv")
        auc_col = "auc"
    else:
        cm = pd.read_csv(data / "table5.csv")
        auc_col = "auc"
    ax.scatter(range(len(cm)), cm[auc_col], color=CORAL, s=28, zorder=3, edgecolor="white", lw=0.5)
    ax.axhline(0.5, color=NEAR, ls=":", lw=0.8)
    ax.axhline(0.7, color=SLATE, ls="--", lw=0.8)
    ax.set_ylim(0.45, 0.75)
    ax.set_xticks([])
    ax.set_xlabel("seven configs")
    ax.set_ylabel("AUC")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "c", "all near chance")

    # (d) rho vs auc
    ax = fig.add_subplot(gs[1, 0])
    if legacy:
        rho = cm.rho50
    else:
        t4 = pd.read_csv(data / "table4.csv")
        t4 = t4[~t4.predictor.isin(["Mean baseline", "Ridge", "chemCPA"])].set_index("predictor")
        rho = cm.set_index("config").loc[t4.index, "auc"].index.map(t4["spearman50"]).reset_index(drop=True)
    ax.scatter(rho, cm[auc_col], color=CORAL, s=28, edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0.5, color=NEAR, ls=":", lw=0.8)
    ax.set_xlabel("Spearman@50")
    ax.set_ylabel("AUC")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "d", "rho != discrimination")

    # (e) bootstrap
    ax = fig.add_subplot(gs[1, 1])
    b = np.load(data / "bootstrap_auc.npy")
    bm = json.load(open(data / "bootstrap_meta.json", encoding="utf-8"))
    ax.hist(b, bins=30, color=SLATE, alpha=0.7)
    ax.axvline(0.5, color=NEAR, ls=":", lw=1)
    ax.axvspan(bm["auc_lo"], bm["auc_hi"], color=CORAL, alpha=0.15)
    ax.set_xlabel("bootstrap AUC")
    ax.set_ylabel("count")
    ax.text(0.5, 0.92, f"95% CI\n[{bm['auc_lo']:.2f},{bm['auc_hi']:.2f}]\nincludes 0.5",
            transform=ax.transAxes, fontsize=6, va="top", ha="center")
    panel(ax, "e", "not significant")

    # (f) gap per config
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(range(len(cm)), cm["gap"], color=CORAL, width=0.6, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xticks([])
    ax.set_xlabel("seven configs")
    ax.set_ylabel("on - off gap")
    ax.set_ylim(0, 0.05)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "f", "gap = 0")

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig4_control_validation.pdf", bbox_inches="tight", dpi=DPI)
    print("  -> fig4_control_validation.pdf done")
    plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
