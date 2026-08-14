"""Graphical abstract for the Applied Soft Computing submission.

Elsevier requires 531 x 1328 px (h x w) or proportionally more, readable at 5 x 13 cm.
Authored at exactly 13 x 5 cm with the tight bbox off, so rendered pt == source pt.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc_context
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT = "/Users/cgxmac/Desktop/CytoBridge/0final/CytoBridge_0707/figs/graphical_abstract.pdf"
CM = 1 / 2.54
GREEN, VERM, PURPLE, INK, GRID, DGRAY = "#009E73", "#D55E00", "#CC79A7", "#222222", "#E9E9E9", "#6E6E6E"
CHANCE, CEILING = 0.500, 0.810

rcParams.update({
    "font.size": 8, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.linewidth": 0.7, "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(13 * CM, 5.2 * CM))
fig.subplots_adjust(left=0.045, right=0.995, top=0.775, bottom=0.235)

# reference lines
ax.axvline(CHANCE, color=DGRAY, ls=":", lw=1.0, zorder=1)
ax.axvline(CEILING, color=INK, ls="--", lw=1.1, zorder=1)

# the band every audited model falls in
ax.axvspan(0.500, 0.542, color=PURPLE, alpha=0.16, lw=0, zorder=0)

MARKS = [
    # x,      y,   colour,  label,                                     dy_text, ha
    (0.542, 0.95, PURPLE, "all four audited models\n0.50 – 0.54", 0.30, "center"),
    (0.588, 1.95, VERM,   "predictor given NO drug\ninformation   0.588", 0.30, "center"),
    (0.717, 0.95, GREEN,  "annotated drug target\n0.717  (70% of range)", 0.30, "center"),
    (0.926, 1.95, GREEN,  "best training compound\n0.926", 0.30, "right"),
]
for x, y, c, lab, dy, ha in MARKS:
    ax.plot([x], [y], "o", color=c, ms=7, zorder=5, mec="white", mew=1.0)
    ax.plot([x, x], [0, y], color=c, lw=1.2, zorder=3, alpha=0.85)
    t = ax.text(x, y + dy, lab, ha=ha, va="bottom", fontsize=7.6, color=INK,
                linespacing=1.25, zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

# structure-based oracles sit on top of chance
ax.plot([0.509, 0.495], [0.52, 0.52], "o", color=VERM, ms=5.5, zorder=5, mec="white", mew=0.8)
t = ax.text(0.500, 0.20, "molecular fingerprint\n0.495 – 0.509", ha="center", va="bottom",
            fontsize=7.2, color=VERM, linespacing=1.2, zorder=6)
t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

ax.text(CHANCE, 2.98, "chance", ha="center", va="bottom", fontsize=7.4, color=DGRAY)
ax.text(CEILING, 2.98, "measured biological ceiling  0.810", ha="center", va="bottom",
        fontsize=7.6, color=INK, fontweight="bold")

ax.set_xlim(0.468, 0.995)
ax.set_ylim(0, 3.30)
ax.set_yticks([])
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_xlabel("Drug-discrimination AUC: does a prediction match its own drug better than another drug?",
              fontsize=8.2, labelpad=3)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)

fig.text(0.5, 0.975, "Per-pair correlation hides that these models cannot tell drugs apart",
         ha="center", va="top", fontsize=9.4, fontweight="bold", color=INK)
fig.text(0.5, 0.885, "The information exists in the data — it is lost in the map from molecular structure to response",
         ha="center", va="top", fontsize=8.0, color=DGRAY)


with rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
    fig.savefig(OUT)
print(f"wrote {OUT}")
