"""Regenerate the README charts from docs/data/*.csv.

    python docs/make_charts.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
DATA = HERE / "data" / "animefilm_l1_val.csv"
OUT = HERE / "img"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
# categorical slots 1-3, fixed order (validated: CVD dE 9.2, normal-vision 27.6)
COLOR = {"FDAT-RD3": "#2a78d6", "FDAT": "#eb6834", "ESRGAN": "#1baf7a"}
ORDER = ["FDAT-RD3", "FDAT", "ESRGAN"]

# Real-world VapourKit fps, 720x540 -> 1440x1080, RTX 4080, TensorRT 11, with encoding.
# FDAT-RD2 is the same network as FDAT-RD3 before the graph rewrite.
SPEED = {"FDAT-RD3": 18.0, "FDAT": 11.0, "ESRGAN": 17.0}

for f in font_manager.findSystemFonts():
    if "Inter" in Path(f).name:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load():
    rows = defaultdict(list)
    with open(DATA) as f:
        for r in csv.DictReader(f):
            rows[r["arch"]].append((int(r["iter"]), float(r["psnr"]), float(r["ssim"]), float(r["topiq"])))
    return rows


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def convergence(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    for ax, (col, label, lo, hi) in zip(axes, ((1, "PSNR (dB)", 25.0, 27.25), (3, "TOPIQ-FR", 0.72, 0.808))):
        style(ax)
        for a in ORDER:
            it = [r[0] / 1000 for r in rows[a]]
            v = [r[col] for r in rows[a]]
            ax.plot(it, v, color=COLOR[a], linewidth=2, label=a, solid_capstyle="round")
        ax.set_xlim(20, 212)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("training iterations (k)")
        ax.set_title(label, loc="left", fontsize=11, fontweight="semibold")
        fmt = "{:.2f}" if col == 1 else "{:.4f}"
        # direct labels at the 200k end, nudged apart, in text ink (colored dot carries identity)
        ends = sorted(((rows[a][-1][col], a) for a in ORDER), reverse=True)
        gap = (hi - lo) * 0.055
        placed = []
        for val, a in ends:
            y = val
            if placed and placed[-1] - y < gap:
                y = placed[-1] - gap
            placed.append(y)
            ax.plot([200], [val], "o", color=COLOR[a], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(fmt.format(val), (200, val), xytext=(203, y), textcoords="data",
                        va="center", fontsize=8.5, color=INK)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.suptitle("ANIMEFILM 2x, L1 only: validation curves (5-image val set)", x=0.06, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "convergence.png")
    plt.close(fig)


def speed_quality(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    style(ax)
    for a in ORDER:
        x, y = SPEED[a], rows[a][-1][1]
        ax.plot([x], [y], "o", color=COLOR[a], markersize=11, markeredgecolor=SURFACE, markeredgewidth=2)
        dx, ha = (0.45, "left") if a != "ESRGAN" else (-0.45, "right")
        ax.annotate(f"{a}\n{y:.2f} dB · {x:g} fps", (x, y), xytext=(x + dx, y), va="center", ha=ha,
                    fontsize=9, color=INK)
    ax.set_xlim(9, 21)
    ax.set_ylim(26.7, 27.2)
    ax.set_xlabel("real-world fps, 720x540 → 1440x1080 (VapourKit, TensorRT, RTX 4080, with encode)")
    ax.set_ylabel("PSNR @ 200k (dB)")
    ax.set_title("Quality vs speed — up and to the right is better", loc="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(OUT / "speed_quality.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    r = load()
    convergence(r)
    speed_quality(r)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
