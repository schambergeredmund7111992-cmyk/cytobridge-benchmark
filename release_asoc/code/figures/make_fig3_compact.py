"""Figure 3 (collapse overview) — compact, Wong-blue. Regenerates fig_collapse_overview.pdf.
Layout-only changes vs the notebook: tighter hspace/wspace, square-filling heatmap cells,
inset colorbars so (b,c,d) stay equal width. Content/numbers unchanged."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# Defaults: data dirs live next to this script (analysis/), figs one level up.
_HERE = Path(__file__).resolve().parent
D1 = sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "data")
D2 = sys.argv[2] if len(sys.argv) > 2 else str(_HERE / "data2")
OUT = sys.argv[3] if len(sys.argv) > 3 else str(_HERE.parent / "figs")
os.makedirs(OUT, exist_ok=True)

# Wong (2011) colorblind-safe palette
BLUE, GREEN, VERM, ORANGE, SKY, PURPLE = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#56B4E9", "#CC79A7"
GRAY, DGRAY, INK, GRID = "#BDBDBD", "#6E6E6E", "#222222", "#EAEAEA"
SEQ = LinearSegmentedColormap.from_list("wong_seq", ["#F2F7FB", "#9CC3DE", "#3E8FC1", "#0072B2", "#023E66"])
DPI = 400
rcParams.update({
  "font.size":7, "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
  "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":0.6, "axes.edgecolor":INK,
  "text.color":INK, "axes.labelcolor":INK, "xtick.color":INK, "ytick.color":INK,
  "axes.labelsize":7.5, "xtick.labelsize":6.5, "ytick.labelsize":6.5,
  "xtick.major.width":0.6, "ytick.major.width":0.6, "xtick.major.size":2.4, "ytick.major.size":2.4,
  "legend.fontsize":6, "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":DPI, "figure.dpi":150,
})

def panel(ax, L, sub=""):
    ax.text(-0.16, 1.16, L, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="right")
    if sub: ax.set_title(sub, fontsize=6.6, loc="center", pad=2.5, color="#555555")

def add_cbar(fig, im, ax, label="", show=True):
    """Append a right-side colorbar axis via make_axes_locatable (robust under
    bbox_inches='tight'). show=False keeps an invisible spacer so sibling
    heatmaps stay exactly equal width."""
    cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.04)
    if not show:
        cax.axis("off"); return None
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    if label: cb.set_label(label, fontsize=6, labelpad=1.5)
    return cb

# Wider first column (bar charts + long y-labels), equal square-ish heatmap columns.
fig = plt.figure(figsize=(7.2, 4.2))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.46,
                       width_ratios=[1.32, 1.0, 1.0, 1.0])

# (a) per-pair Spearman@50: no configuration beats Mean
ax = fig.add_subplot(gs[0, 0]); t1 = pd.read_csv(f"{D1}/table1_regenerated.csv")
cb = t1[~t1.predictor.str.contains("Ridge|chemCPA|Mean", case=False, na=False)].sort_values("spearman50_ondiag")
labs = list(cb.predictor) + ["Ridge", "chemCPA"]; vals = list(cb.spearman50_ondiag) + [0.331, 0.086]
colors = [BLUE]*len(cb) + [DGRAY, GRAY]
ax.barh(range(len(labs)), vals, color=colors, height=0.74, edgecolor="white", lw=0.4, zorder=3)
mean_rho = float(t1.loc[t1.predictor.str.startswith("Mean"), "spearman50_ondiag"].iloc[0])
ax.axvline(mean_rho, color=GREEN, lw=1.3, zorder=4)
ax.text(mean_rho - 0.024, len(labs)/2.0, f"Mean = {mean_rho:.3f}", ha="left", va="center", rotation=90, fontsize=6, color=GREEN, fontweight="bold")
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.8)
ax.set_xlabel("Spearman@50 (on-diagonal)"); ax.set_xlim(0, max(0.45, mean_rho + 0.06))
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=BLUE, label="CytoBridge"), Patch(color=DGRAY, label="Ridge"), Patch(color=GRAY, label="chemCPA")],
          fontsize=5.2, frameon=False, loc="lower right", handlelength=1.0, handleheight=1.0, labelspacing=0.25, borderpad=0.2)
panel(ax, "a")

# (b-d) inter-drug r heatmaps per cell line, shared colorbar on (d).
# Invisible spacer cax on (b),(c) keeps all three heatmaps exactly equal width.
for k, (cpos, cell) in enumerate([(gs[0, 1], "A549"), (gs[0, 2], "K562"), (gs[0, 3], "MCF7")]):
    ax = fig.add_subplot(cpos); Mc = np.load(f"{D2}/interdrug_{cell}.npy")
    im = ax.imshow(Mc, vmin=0.9, vmax=1.0, cmap=SEQ, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); panel(ax, "bcd"[k], cell)
    add_cbar(fig, im, ax, "inter-drug $r$", show=(cell == "MCF7"))

# (e) per-cell mean inter-drug r
ax = fig.add_subplot(gs[1, 0]); pc = pd.read_csv(f"{D2}/per_cellline.csv").groupby("cell").inter_drug_r.mean()
ax.bar(range(len(pc)), pc.values, color=BLUE, width=0.62, edgecolor="white", lw=0.5, zorder=3)
ax.set_xticks(range(len(pc))); ax.set_xticklabels(pc.index, fontsize=6.5); ax.set_ylim(0.9, 1.0)
ax.set_ylabel("mean inter-drug $r$"); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); panel(ax, "e")

# (f) per-gene s.d. across drugs, predicted vs true
ax = fig.add_subplot(gs[1, 1]); gv = np.load(f"{D2}/gene_variance.npz")
ax.hist(gv["true_std"], bins=40, color=GREEN, alpha=0.65, label="true", log=True, edgecolor="none")
ax.hist(gv["pred_std"], bins=40, color=VERM, alpha=0.6, label="predicted", log=True, edgecolor="none")
ax.set_xlabel("per-gene s.d. across drugs"); ax.set_ylabel("genes")
ax.legend(fontsize=5.8, frameon=False, loc="upper right"); panel(ax, "f")

# (g) matched vs mismatched correlation distribution
ax = fig.add_subplot(gs[1, 2]); oo = pd.read_csv(f"{D1}/onoff_loss-only.csv")
p = ax.violinplot([oo[oo.kind == "on"].score, oo[oo.kind == "off"].score], showmeans=True, widths=0.82)
for b, col in zip(p["bodies"], [BLUE, GRAY]): b.set_facecolor(col); b.set_alpha(0.65); b.set_edgecolor(col); b.set_linewidth(0.6)
for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
    if key in p: p[key].set_color(INK); p[key].set_linewidth(0.7)
ax.set_xticks([1, 2]); ax.set_xticklabels(["matched", "mismatched"], fontsize=6); ax.set_ylabel("top-50 correlation"); panel(ax, "g")

# (h) drug-identity confusion
ax = fig.add_subplot(gs[1, 3]); A = np.load(f"{D2}/confusion_A549.npy")
im = ax.imshow(A, cmap=SEQ, vmin=0, vmax=0.4, aspect="equal")
ax.set_xlabel("assigned drug"); ax.set_ylabel("true drug"); ax.set_xticks([]); ax.set_yticks([])
add_cbar(fig, im, ax, "P(assign)", show=True); panel(ax, "h")

fig.savefig(f"{OUT}/fig_collapse_overview.pdf", bbox_inches="tight", dpi=DPI)
print("fig3 done ->", f"{OUT}/fig_collapse_overview.pdf")
