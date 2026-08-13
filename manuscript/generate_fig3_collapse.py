"""Figure 3: Collapse overview (8 panels)."""
import numpy as np, pandas as pd, json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent
D1 = ROOT / "analysis" / "data"
D2 = ROOT / "analysis" / "data2"
OUT = ROOT / "figs"
os.makedirs(OUT, exist_ok=True)

TEAL, CORAL, SLATE, NEAR, GRID, GOLD = "#2A9D8F", "#E76F51", "#6C7A89", "#1F2937", "#E5E7EB", "#E9C46A"
DPI = 300
rcParams.update({"font.size":8,"font.family":"sans-serif","axes.spines.top":False,
  "axes.spines.right":False,"axes.linewidth":0.7,"pdf.fonttype":42,"axes.edgecolor":NEAR,
  "text.color":NEAR,"axes.labelcolor":NEAR,"xtick.color":NEAR,"ytick.color":NEAR,
  "xtick.labelsize":7,"ytick.labelsize":7})

def panel(ax, L, title=""):
    ax.set_title(f"{title}", fontsize=8.5, loc="left", pad=3)
    ax.text(-0.02, 1.16, L, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")

print("Generating Figure 3: collapse overview...")
fig = plt.figure(figsize=(7.2, 5.4))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.62, wspace=0.5)
cm = pd.read_csv(f"{D1}/config_metrics.csv").sort_values("rho50")

# (a) Spearman vs Mean
ax = fig.add_subplot(gs[0, 0])
labs = list(cm.config) + ["Ridge"]
vals = list(cm.rho50) + [0.312]
ax.barh(range(len(labs)), vals, color=[CORAL]*len(cm)+[SLATE], height=0.7, edgecolor="white", lw=0.4, zorder=3)
ax.axvline(0.357, color=TEAL, lw=1.5)
ax.set_yticks(range(len(labs)))
ax.set_yticklabels(labs, fontsize=6)
ax.set_xlabel("Spearman@50")
ax.set_xlim(0, 0.4)
ax.xaxis.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
panel(ax, "a", "none beats Mean")

# (b-d) inter-drug heatmaps per cell line
for k, (c, cell) in enumerate([(gs[0, 1], "A549"), (gs[0, 2], "K562"), (gs[0, 3], "MCF7")]):
    ax = fig.add_subplot(c)
    Mc = np.load(f"{D2}/interdrug_{cell}.npy")
    im = ax.imshow(Mc, vmin=0.9, vmax=1.0, cmap="magma")
    ax.set_xticks([])
    ax.set_yticks([])
    panel(ax, "bcd"[k], f"inter-drug r, {cell}")

# (e) per-cell inter-drug r
ax = fig.add_subplot(gs[1, 0])
pc = pd.read_csv(f"{D2}/per_cellline.csv").groupby("cell").inter_drug_r.mean()
ax.bar(range(len(pc)), pc.values, color=CORAL, width=0.6, edgecolor="white", lw=0.5, zorder=3)
ax.set_xticks(range(len(pc)))
ax.set_xticklabels(pc.index, fontsize=7)
ax.set_ylim(0.9, 1.0)
ax.set_ylabel("mean inter-drug r")
ax.yaxis.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
panel(ax, "e", "collapse in every cell")

# (f) per-gene std pred vs true
ax = fig.add_subplot(gs[1, 1])
gv = np.load(f"{D2}/gene_variance.npz")
ax.hist(gv["true_std"], bins=40, color=TEAL, alpha=0.6, label="true", log=True)
ax.hist(gv["pred_std"], bins=40, color=CORAL, alpha=0.6, label="predicted", log=True)
ax.set_xlabel("per-gene std across drugs")
ax.set_ylabel("genes")
ax.legend(fontsize=6, frameon=False)
panel(ax, "f", "predictions are flat")

# (g) on/off violin
ax = fig.add_subplot(gs[1, 2])
oo = pd.read_csv(f"{D1}/onoff_loss-only.csv")
p = ax.violinplot([oo[oo.kind == "on"].score, oo[oo.kind == "off"].score], showmeans=True)
for b, col in zip(p["bodies"], [TEAL, CORAL]):
    b.set_facecolor(col)
    b.set_alpha(0.6)
ax.set_xticks([1, 2])
ax.set_xticklabels(["on", "off"], fontsize=7)
ax.set_ylabel("correlation")
panel(ax, "g", "on = off")

# (h) confusion matrix
ax = fig.add_subplot(gs[1, 3])
A = np.load(f"{D2}/confusion_A549.npy")
im = ax.imshow(A, cmap="magma", vmin=0, vmax=0.4)
ax.set_xlabel("assigned drug")
ax.set_ylabel("true drug")
ax.set_xticks([])
ax.set_yticks([])
panel(ax, "h", "identity at chance")

fig.savefig(f"{OUT}/fig3_collapse_overview.pdf", bbox_inches="tight", dpi=DPI)
print("  -> fig3_collapse_overview.pdf done")
plt.close('all')
