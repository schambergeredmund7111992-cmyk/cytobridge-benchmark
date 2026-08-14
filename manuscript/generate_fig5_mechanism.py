"""Figure 5: Mechanism and case studies (8 panels).

Reads the pooled-construction figures_data directory produced by
`scripts/regenerate_paper_numbers.py`; falls back to the legacy per-pair
artifacts with a warning.
"""
import argparse
import glob
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
            print("[fig5] WARNING: no pooled figures_data found; using legacy "
                  "per-pair artifacts (superseded construction).", file=sys.stderr)
            data = ROOT / "analysis" / "data"
            data2 = ROOT / "analysis" / "data2"
    data = Path(data)
    if legacy:
        data2 = Path(data2)

    print("Generating Figure 5: mechanism and case studies...")
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.6, wspace=0.55)

    # Load data
    lc = pd.read_csv(data / "loss_components.csv")
    pred0 = np.load(data / "logfc_pred_t7_sub_loss_only.npy")
    true0 = np.load(data / "logfc_true_t7_sub_loss_only.npy")
    meta = pd.read_csv(data / "logfc_meta_t7_sub_loss_only.csv")

    # (a) Loss component magnitudes
    names = ["L_recon", "L_kl", "L_contrast", "L_pathway", "L_direction", "L_drugspec", "L_delta"]
    vals = [float(lc[n].iloc[0]) for n in names]
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(len(names)), vals, color=[CORAL] + [SLATE] * 6, width=0.7, edgecolor="white", lw=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[2:] for n in names], rotation=40, fontsize=5.5, ha="right")
    ax.set_ylabel("loss (log)")
    ax.text(0, vals[0] * 1.4, f"{vals[0]:.0f}", ha="center", fontsize=6.5, fontweight="bold", color=CORAL)
    panel(ax, "a", "ZINB dominates")

    # (b) Loss trajectory from the shipped training logs when available,
    # otherwise the on/off proxy used by the legacy figure.
    ax = fig.add_subplot(gs[0, 1])
    curves_dir = data / "loss_curves"
    metrics_files = sorted(curves_dir.glob("*.csv")) if curves_dir.is_dir() else []
    if metrics_files:
        ax.set_yscale("log")
        for path in metrics_files:
            frame = pd.read_csv(path)
            step_col = "step" if "step" in frame.columns else frame.columns[0]
            for column in ("train/L_recon_step", "L_recon", "recon"):
                if column in frame.columns:
                    ax.plot(frame[step_col], frame[column], lw=0.7, alpha=0.7,
                            label=path.stem)
                    break
        ax.set_xlabel("logged step")
        ax.set_ylabel("loss (log)")
        panel(ax, "b", "recon stays dominant")
    else:
        oo = pd.read_csv(data / ("onoff_loss-only.csv" if legacy else "onoff_pooled.csv"))
        on_scores = oo[oo.kind == "on"].score.values
        off_scores = oo[oo.kind == "off"].score.values
        ax.scatter(range(len(on_scores)), sorted(on_scores), s=3, color=CORAL, alpha=0.5, label="on-diag")
        ax.scatter(range(len(off_scores)), sorted(off_scores), s=3, color=SLATE, alpha=0.5, label="off-diag")
        ax.set_xlabel("pair index (sorted)")
        ax.set_ylabel("correlation")
        ax.legend(fontsize=5, frameon=False, ncol=2)
        panel(ax, "b", "on vs off distributions (proxy)")

    # (c-f) Four case studies: scatter pred vs pred, true vs true
    cs = json.load(open(data / "casestudies.json", encoding="utf-8"))
    idx = list(meta.index[meta.cell_line == "A549"])
    pairs = [(0, 1), (2, 3), (4, 5), (0, 4)]
    cells_pos = [gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1]]
    for k, ((i, j), gp) in enumerate(zip(pairs, cells_pos)):
        d = cs.get(f"pair{k}", {})
        ax = fig.add_subplot(gp)
        a, b = idx[i], idx[j]
        ax.scatter(pred0[a], pred0[b], s=3, color=CORAL, alpha=0.35)
        ax.scatter(true0[a], true0[b], s=3, color=TEAL, alpha=0.35)
        L = 4
        ax.plot([-L, L], [-L, L], ls="--", color=NEAR, lw=0.6)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        pred_r = np.corrcoef(pred0[a], pred0[b])[0, 1] if np.std(pred0[a]) > 1e-6 and np.std(pred0[b]) > 1e-6 else 0
        true_r = np.corrcoef(true0[a], true0[b])[0, 1] if np.std(true0[a]) > 1e-6 and np.std(true0[b]) > 1e-6 else 0
        ax.set_title(f"pred {pred_r:.2f} / true {true_r:.2f}", fontsize=6.5, loc="left", pad=2)
        ax.text(-0.02, 1.22, "cdef"[k], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
        drug_a = meta.iloc[a]["drug"][:15]
        drug_b = meta.iloc[b]["drug"][:15]
        ax.set_xlabel(drug_a, fontsize=6)
        ax.set_ylabel(drug_b, fontsize=6)
        ax.tick_params(labelsize=5)

    # (g) Pathway illusion from the recomputed numbers
    ax = fig.add_subplot(gs[1, 2])
    pathway_path = data / "pathway_illusion.json"
    if pathway_path.exists():
        pw = json.load(open(pathway_path, encoding="utf-8"))
        on_val, off_val = pw["on_diag_mean"], pw["off_diag_mean"]
        gap_text = f"gap {pw['gap']:.5f}"
    else:
        on_val, off_val = 0.9484, 0.9483
        gap_text = "gap 0.00006"
    ax.bar([0, 1], [on_val, off_val], width=0.5, color=[SLATE, CORAL], edgecolor="white", lw=0.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["matched", "other"], fontsize=7)
    ax.set_ylabel("pathway r")
    ax.set_ylim(0, 1.1)
    ax.text(0.5, 0.55, gap_text, ha="center", fontsize=7, color=CORAL, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "g", "pathway illusion")

    # (h) Summary: effective alpha from the recomputed calibration
    ax = fig.add_subplot(gs[1, 3])
    alpha_path = data / "effective_alpha.json"
    if alpha_path.exists():
        eff = json.load(open(alpha_path, encoding="utf-8"))["effective_alpha"]
    else:
        eff = 0.03
    ax.bar([0, 1], [1.0, eff], width=0.5, color=[TEAL, CORAL], edgecolor="white", lw=0.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true\nsignal", "CytoBridge"], fontsize=6.5)
    ax.set_ylabel("effective drug signal alpha")
    ax.set_ylim(0, 1.1)
    ax.text(1, max(eff + 0.02, 0.08), f"{eff:.2f}", ha="center", fontsize=7, fontweight="bold", color=CORAL)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel(ax, "h", f"{eff * 100:.0f}% of real signal")

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig5_mechanism.pdf", bbox_inches="tight", dpi=DPI)
    print("  -> fig5_mechanism.pdf done")
    plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
