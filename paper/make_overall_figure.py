"""
Draw the pipeline figure referenced as figure/Overall.png.

Rewritten 2026-09-13 to show the overall vision as the four things the method
actually is:
  (1) the problem expressed as a weapon-target graph and embedded by a GNN,
  (2) reinforcement learning deciding which weapons fire,
  (3) the auction deciding what each firing weapon engages,
  (4) refinement of the completed schedule.

(1) draws a real bipartite graph rather than a box labelled "encoder", because
the graph representation is part of the claim, not an implementation detail.
Steps (1) to (3) repeat once per stage; (4) runs once, over the whole
schedule, which is why it sits on its own row.

Grayscale-safe: fills are light grays, emphasis is border weight, not hue.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "Overall.png")

FILL_LEARNED = "#d2d2d2"
FILL_FIXED = "#f5f5f5"
EDGE = "#222222"
MUTED = "#555555"
NL = chr(10)

fig, ax = plt.subplots(figsize=(9.2, 3.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.3)
ax.axis("off")


def box(x, y, w, h, learned=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                linewidth=2.0 if learned else 1.0,
                                edgecolor=EDGE,
                                facecolor=FILL_LEARNED if learned else FILL_FIXED,
                                zorder=2))


def label(x, y, text, fontsize=9.0, weight="normal"):
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center",
            zorder=3, linespacing=1.35, fontweight=weight)


def step_no(x, y, n):
    ax.text(x, y, "(%d)" % n, fontsize=8.6, ha="center", va="center",
            color=MUTED, zorder=3, style="italic")


def arrow(x1, y1, x2, y2, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, linestyle=ls,
                                 color=EDGE, zorder=4))


Y, H = 2.20, 1.45

# ---------------------------------------------- (1) problem as a graph + GNN
box(0.30, Y, 2.85, H)
step_no(0.52, Y + H - 0.18, 1)

# a small bipartite weapon-target graph drawn inside the box
wx, tx = 1.00, 2.05
wys = [Y + 1.02, Y + 0.74, Y + 0.46]
tys = [Y + 1.02, Y + 0.74, Y + 0.46]
for wy in wys:
    for ty in tys:
        ax.plot([wx, tx], [wy, ty], color="#999999", lw=0.6, zorder=2.5)
for wy in wys:
    ax.add_patch(Circle((wx, wy), 0.085, facecolor="white", edgecolor=EDGE,
                        lw=1.0, zorder=3))
for ty in tys:
    ax.add_patch(Circle((tx, ty), 0.085, facecolor="#bbbbbb", edgecolor=EDGE,
                        lw=1.0, zorder=3))
ax.text(wx, Y + 1.20, "weapons", fontsize=7.8, ha="center", va="center",
        color=MUTED, zorder=3)
ax.text(tx, Y + 1.20, "targets", fontsize=7.8, ha="center", va="center",
        color=MUTED, zorder=3)
label(1.72, Y + 0.20, "GNN embedding", fontsize=9.0)

# ------------------------------------------------------ (2) learned decision
box(3.55, Y, 2.35, H, learned=True)
step_no(3.77, Y + H - 0.18, 2)
label(4.72, Y + 0.80, "Reinforcement" + NL + "learning decides" + NL
      + "which weapons fire")
label(4.72, Y + 0.28, "$a_{m,t} \\in \\{0,1\\}$", fontsize=9.4)

# -------------------------------------------------------------- (3) auction
box(6.30, Y, 2.35, H)
step_no(6.52, Y + H - 0.18, 3)
label(7.47, Y + 0.80, "Auction decides" + NL + "what each firing" + NL
      + "weapon engages")
label(7.47, Y + 0.28, "no learned parameters", fontsize=8.2)

arrow(3.15, Y + H / 2, 3.55, Y + H / 2)
arrow(5.90, Y + H / 2, 6.30, Y + H / 2)

ax.text(5.0, 4.05, "Repeated once per stage $t = 1 \\ldots T$",
        fontsize=9.2, style="italic", color=MUTED, ha="center", va="center")

# next-stage feedback, routed below the row through empty space
lane = Y - 0.45
ax.plot([7.80, 7.80], [Y, lane], color=EDGE, lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.plot([7.80, 1.30], [lane, lane], color=EDGE, lw=1.0, ls=(0, (4, 2)), zorder=1)
arrow(1.30, lane, 1.30, Y, lw=1.0, ls=(0, (4, 2)))
ax.text(3.30, lane - 0.19, "apply the shots, then next stage", fontsize=8.4,
        style="italic", color=MUTED, ha="center", va="center")

# ----------------------------------------------------------- (4) refinement
arrow(8.85, Y - 0.02, 8.85, 1.02, lw=1.4)
ax.text(8.70, 1.30, "completed schedule", fontsize=8.4, ha="right",
        va="center", color=EDGE)

box(0.30, 0.20, 9.40, 0.82)
step_no(0.52, 0.86, 4)
label(5.00, 0.61, "Refinement: flip one stage-weapon decision, re-simulate, "
      "keep it only if the objective improves." + NL
      + "Repeated up to $K$ times. No network call anywhere in this loop.")

# ------------------------------------------------------------------- legend
lx, ly = 0.30, 3.97
ax.add_patch(FancyBboxPatch((lx, ly), 0.26, 0.16,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            linewidth=2.0, edgecolor=EDGE,
                            facecolor=FILL_LEARNED, zorder=5))
ax.text(lx + 0.36, ly + 0.08, "learned", fontsize=8.4, va="center", zorder=5)
ax.add_patch(FancyBboxPatch((lx + 1.25, ly), 0.26, 0.16,
                            boxstyle="round,pad=0.02,rounding_size=0.04",
                            linewidth=1.0, edgecolor=EDGE,
                            facecolor=FILL_FIXED, zorder=5))
ax.text(lx + 1.61, ly + 0.08, "fixed", fontsize=8.4, va="center", zorder=5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote %s" % OUT)
