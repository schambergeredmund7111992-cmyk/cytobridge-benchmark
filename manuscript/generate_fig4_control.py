"""Figure 4: Control validation (6 panels)."""
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

print("Generating Figure 4: control validation...")
fig = plt.figure(figsize=(7.2, 4.2))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)

# (a) calibration curve
ax = fig.add_subplot(gs[0, 0])
cal = pd.read_csv(f"{D2}/calibration.csv")
meta = json.load(open(f"{D2}/calibration_meta.json"))
ax.plot(cal.alpha, cal.auc, "-o", color=TEAL, ms=3, lw=1.4, zorder=3)
ea = meta["effective_alpha"]
ax.axhline(0.583, color=CORAL, ls="--", lw=1)
ax.axvline(ea, color=CORAL, ls="--", lw=1)
ax.scatter([ea], [0.583], color=CORAL, s=30, zorder=4)
ax.text(ea+0.04, 0.60, f"CytoBridge\nalpha={ea:.2f}", fontsize=6.5, color=CORAL)
ax.set_xlabel("injected drug signal alpha")
ax.set_ylabel("discrimination AUC")
ax.set_ylim(0.45, 1.03)
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
panel(ax, "a", "control tracks real signal")

# (b) ladder
ax = fig.add_subplot(gs[0, 1])
v = pd.read_csv(f"{D1}/control_validation.csv").set_index("predictor")
order_names = ["Random (50 perm)", "Mean (collapsed)", "Ridge (clean)", "CytoBridge (best)", "Oracle (truth)"]
# Check which predictors exist
available = [n for n in order_names if n in v.index]
vals = [v.loc[n, "auc"] for n in available]
labels_short = ["Rand", "Mean", "Ridge", "CB", "Oracle"][:len(available)]
colors_bar = [SLATE, SLATE, SLATE, CORAL, TEAL][:len(available)]
ax.bar(range(len(available)), vals, color=colors_bar, width=0.66, edgecolor="white", lw=0.5, zorder=3)
ax.axhline(0.5, color=NEAR, ls=":", lw=0.8)
ax.axhline(0.7, color=SLATE, ls="--", lw=0.8)
ax.set_xticks(range(len(available)))
ax.set_xticklabels(labels_short, fontsize=6)
ax.set_ylim(0, 1.08)
ax.set_ylabel("AUC")
for i, a in enumerate(vals):
    ax.text(i, a+0.02, f"{a:.2f}", ha="center", fontsize=6.5, fontweight="bold")
ax.yaxis.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
panel(ax, "b", "calibrated 0.5 to 1.0")

# (c) per-config AUC
ax = fig.add_subplot(gs[0, 2])
cm = pd.read_csv(f"{D1}/config_metrics.csv")
ax.scatter(range(len(cm)), cm.auc, color=CORAL, s=28, zorder=3, edgecolor="white", lw=0.5)
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
ax.scatter(cm.rho50, cm.auc, color=CORAL, s=28, edgecolor="white", lw=0.5, zorder=3)
ax.axhline(0.5, color=NEAR, ls=":", lw=0.8)
ax.set_xlabel("Spearman@50")
ax.set_ylabel("AUC")
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
panel(ax, "d", "rho != discrimination")

# (e) bootstrap
ax = fig.add_subplot(gs[1, 1])
b = np.load(f"{D2}/bootstrap_auc.npy")
bm = json.load(open(f"{D2}/bootstrap_meta.json"))
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

fig.savefig(f"{OUT}/fig4_control_validation.pdf", bbox_inches="tight", dpi=DPI)
print("  -> fig4_control_validation.pdf done")
plt.close('all')
