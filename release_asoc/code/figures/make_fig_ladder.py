"""Rebuild fig_ladder.pdf: the oracle ladder, with recovered fractions on the corrected scale.

Authored at final size (7.0 in two-column) with the tight bbox switched off, so source pt
equals rendered pt when inserted at \\linewidth.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc_context
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT = "/Users/cgxmac/Desktop/CytoBridge/0final/CytoBridge_0707/figs/fig_ladder.pdf"
GREEN, VERM, PURPLE, INK, GRID, DGRAY = "#009E73", "#D55E00", "#CC79A7", "#222222", "#EAEAEA", "#6E6E6E"
CEILING, CHANCE = 0.810, 0.500
RANGE = CEILING - CHANCE

# label, AUC, kind
ROWS = [
    ("Oracle (positive control)",                   1.000, "anchor"),
    ("Retrieval upper bound (hindsight)",           0.926, "oracle"),
    ("Cross-plate biological replicate",            0.810, "ceiling"),
    ("Target-matched oracle (15/27 anchors)",       0.717, "oracle"),
    ("CytoBridge (best configuration)",             0.5417, "audited"),
    ("Ridge (as audited)",                          0.510, "audited"),
    ("Chemical 1-NN oracle",                        0.509, "structure"),
    ("Random (negative control)",                   0.500, "anchor"),
    ("chemCPA / biolord (as audited)",              0.500, "audited"),
    ("Mean (negative control)",                     0.500, "anchor"),
    ("Morgan-Ridge oracle (all 160 compounds)",     0.495, "structure"),
]
COLOR = {"oracle": GREEN, "ceiling": INK, "audited": PURPLE, "structure": VERM, "anchor": DGRAY}
NO_PHI = {"Oracle (positive control)", "Target-matched oracle (15/27 anchors)"}   # target oracle 用 15/27 个 anchor,不可归一化

rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.linewidth": 0.7, "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.0, 3.9))
fig.subplots_adjust(left=0.325, right=0.985, top=0.985, bottom=0.135)
y = list(range(len(ROWS)))[::-1]

ax.axvline(CHANCE, color=DGRAY, ls=":", lw=0.9, zorder=1)
ax.axvline(CEILING, color=INK, ls="--", lw=1.0, zorder=1)
ax.text(CHANCE, len(ROWS) - 0.35, "chance", fontsize=7.5, color=DGRAY, ha="center", va="bottom")
ax.text(CEILING, len(ROWS) - 0.35, "achievable ceiling", fontsize=7.5, color=INK, ha="center", va="bottom")

for yi, (label, auc, kind) in zip(y, ROWS):
    c = COLOR[kind]
    ax.plot([CHANCE, auc], [yi, yi], color=c, lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot([auc], [yi], "o", color=c, ms=6.5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)
    txt = f"{auc:.3f}"
    if label not in NO_PHI:
        txt += f"   {round((auc - CHANCE) / RANGE * 100):d}%"
    t = ax.text(auc + 0.012, yi, txt, va="center", ha="left", fontsize=8, color=INK,
                fontweight="bold" if kind in ("audited", "ceiling") else "normal", zorder=5)
    t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.478, 1.075)
ax.set_ylim(-0.7, len(ROWS) - 0.1)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Drug-discrimination AUC   (% = fraction of the biological ceiling recovered)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)

ax.legend(handles=[
    Line2D([], [], color=GREEN, marker="o", ls="-", ms=5.5, lw=1.6, label="oracle given measured responses"),
    Line2D([], [], color=INK, marker="o", ls="-", ms=5.5, lw=1.6, label="measured biological ceiling"),
    Line2D([], [], color=PURPLE, marker="o", ls="-", ms=5.5, lw=1.6, label="audited predictor"),
    Line2D([], [], color=VERM, marker="o", ls="-", ms=5.5, lw=1.6, label="structure-based oracle"),
    Line2D([], [], color=DGRAY, marker="o", ls="-", ms=5.5, lw=1.6, label="metric anchor"),
], fontsize=7.6, frameon=False, loc="lower right", ncol=2, handlelength=1.5,
    labelspacing=0.3, columnspacing=1.1, borderpad=0.2)

with rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
    fig.savefig(OUT)
print(f"wrote {OUT}")
for label, auc, _ in ROWS:
    phi = "" if label in NO_PHI else f"{round((auc - CHANCE) / RANGE * 100):d}%"
    print(f"  {label:42s} {auc:.3f}  {phi}")
