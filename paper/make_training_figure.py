"""
Draw the training figure referenced as figure/Training.png.

Rewritten 2026-09-13. The previous version showed two policies being trained
against each other; the reported method trains one, and trains it THROUGH a
fixed search rather than on its own raw output. That is the single idea this
figure has to carry.

Layout: P parallel rollouts of one instance fan out, each is searched, each
returns a post-search objective, and those returns are standardized against
each other to form the advantage. No critic appears anywhere, which is the
point of the empty space where one would normally sit.

Routing rule followed throughout: feedback edges run in lanes that contain no
boxes. Arcing them across a row makes them cross everything in between.

Grayscale-safe: fills are light grays, emphasis is border weight, not hue.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure", "Training.png")

FILL_LEARNED = "#d2d2d2"
FILL_FIXED = "#f5f5f5"
EDGE = "#222222"
NL = chr(10)

fig, ax = plt.subplots(figsize=(9.6, 5.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.0)
ax.axis("off")


def box(x, y, w, h, text, learned=False, fontsize=8.6, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                linewidth=2.0 if learned else 1.0,
                                linestyle="--" if dashed else "-",
                                edgecolor=EDGE,
                                facecolor=FILL_LEARNED if learned else FILL_FIXED,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize, ha="center",
            va="center", zorder=3, linespacing=1.35)


def arrow(x1, y1, x2, y2, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, linestyle=ls,
                                 color=EDGE, zorder=4))


def dash(pts, lw=1.2):
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        ax.plot([x1, x2], [y1, y2], color=EDGE, lw=lw, ls=(0, (4, 2)), zorder=1)


# ------------------------------------------------------------------ instance
box(0.25, 2.85, 1.50, 0.90, "One training" + NL + "instance", fontsize=8.6)

# ------------------------------------------------------- P parallel rollouts
ROWS = [4.35, 3.00, 1.65]          # box bottoms
BH = 0.80
for i, y in enumerate(ROWS):
    dashed = (i == 2)
    box(2.45, y, 1.80, BH, "Sample fire/hold," + NL + "auction, step",
        fontsize=8.2, dashed=dashed)
    box(4.70, y, 1.60, BH, "Fixed search," + NL + "$K_{\\mathrm{train}}$ edits",
        fontsize=8.2, dashed=dashed)
    box(6.75, y, 1.15, BH, "$R^{(%s)}$" % ("p" if i == 2 else str(i + 1)),
        fontsize=9.2, dashed=dashed)
    arrow(1.75, 3.30, 2.45, y + BH / 2, lw=1.0)
    arrow(4.25, y + BH / 2, 4.70, y + BH / 2, lw=1.0)
    arrow(6.30, y + BH / 2, 6.75, y + BH / 2, lw=1.0)

ax.text(3.35, 2.63, "$\\vdots$", fontsize=13, ha="center", va="center",
        color="#555555")
ax.text(5.20, 5.38, "$P$ parallel rollouts of the same instance", fontsize=8.6,
        style="italic", color="#555555", ha="center", va="center")

# ------------------------------------------------------------------ advantage
box(8.35, 2.85, 1.35, 0.90,
    "Standardize" + NL + "across $P$" + NL + "$\\rightarrow\\ \\hat{A}$", fontsize=8.2)
for y in ROWS:
    arrow(7.90, y + BH / 2, 8.35, 3.30, lw=1.0)

ax.text(9.02, 2.28, "no critic anywhere", fontsize=8.4, style="italic",
        color="#555555", ha="center", va="center")

# ------------------------------------------------------- gradient, lane y=0.62
box(3.30, 0.25, 3.35, 0.78,
    "REINFORCE on the binary event only:" + NL
    + "$\\nabla_\\theta \\log \\pi_m(a_{m,t})$ weighted by $\\hat{A}$",
    learned=True, fontsize=8.0)

# advantage -> update: down the right margin, then left into the box edge
dash([(9.02, 2.85), (9.02, 0.64), (6.70, 0.64)])
arrow(6.75, 0.64, 6.65, 0.64, lw=1.2, ls=(0, (4, 2)))

# update -> policy: left out of the box, then up into the rollout column
dash([(3.30, 0.64), (2.90, 0.64), (2.90, 1.60)])
arrow(2.90, 1.55, 2.90, 1.65, lw=1.2, ls=(0, (4, 2)))

# ------------------------------------------------------------------ callout
ax.text(5.00, 5.82,
        "the return credited to the policy is measured after the search, so the policy",
        fontsize=8.6, ha="center", va="center", color="#222222")
ax.text(5.00, 5.60,
        "is optimized for what the search makes of its decisions",
        fontsize=8.6, ha="center", va="center", color="#222222")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote %s" % OUT)
