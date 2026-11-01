"""Draw docs/img/windows.png: the alternating 10x30 / 30x10 attention windows.

    python docs/make_windows_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent / "img" / "windows.png"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE = "#2a78d6", "#eb6834"

for f in font_manager.findSystemFonts():
    if "Inter" in Path(f).name:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

H, W = 60, 90          # a 60x90 patch of the body's feature map (body pixels)
PX = (35, 44)          # the query pixel (row, col)


def panel(ax, title, wins, colors):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor=GRID, linewidth=1))
    for (wh, ww), color in zip(wins, colors):
        for r in range(0, H, wh):
            for c in range(0, W, ww):
                ax.add_patch(Rectangle((c, r), ww, wh, fill=False, edgecolor=GRID, linewidth=0.8))
        r0, c0 = PX[0] // wh * wh, PX[1] // ww * ww
        ax.add_patch(Rectangle((c0, r0), ww, wh, facecolor=color, alpha=0.28, edgecolor=color, linewidth=2))
    ax.add_patch(Rectangle((PX[1], PX[0]), 1.6, 1.6, facecolor=INK, edgecolor="none"))
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)


fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), dpi=160)
panel(axes[0], "Spatial block A: 10×30 windows", [(10, 30)], [BLUE])
panel(axes[1], "Spatial block B: 30×10 windows", [(30, 10)], [ORANGE])
panel(axes[2], "What one pixel has seen after A + B", [(10, 30), (30, 10)], [BLUE, ORANGE])
fig.text(0.012, 0.03, "Each window is 300 tokens. The black square is one body pixel; shaded is what it attends to. "
         "Consecutive spatial blocks swap orientation, so lines running either way get long-range context.",
         fontsize=8.5, color=INK_2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT)
print("wrote", OUT)
