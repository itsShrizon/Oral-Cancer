"""fig01c — AttentionHub v1 (parallel fusion) vs v2 (sequential cascade) — clean."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _lib.style import (apply_rc, save_fig, COLOR_HUB_FILL, COLOR_HUB_EDGE,
                        COLOR_ATTN_FILL, COLOR_ATTN_EDGE,
                        COLOR_IO_FILL, COLOR_IO_EDGE, PROPOSED_COLOR,
                        PROPOSED_EDGE, FAIL_RED, INK, MUTED)
from _lib.layout import add_block, add_arrow, setup_canvas


def draw_v1(ax, x0):
    ax.text(x0 + 40, 129, "AttentionHub-v1  —  Parallel Fusion",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#9A6418")

    add_block(ax, (x0 + 27, 118), 26, 9, title="Input  (96 x 14 x 14)",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=11)

    # Three branches
    bx = [x0 + 4, x0 + 30, x0 + 56]
    branches = [("BAM",     "spatial + channel"),
                ("Triplet", "cross-dimensional"),
                ("KAN",     "B-spline channel")]
    for x, (name, sub) in zip(bx, branches):
        add_block(ax, (x, 102), 24, 9,
                  title="Conv 1x1   96 -> 48", sub="reduce",
                  facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=10.5)
        add_block(ax, (x, 82), 24, 14, title=name, sub=sub,
                  facecolor=COLOR_ATTN_FILL, edgecolor=COLOR_ATTN_EDGE,
                  fontsize_title=12, fontsize_sub=10)
        add_arrow(ax, (x0 + 40, 118), (x + 12, 111), lw=1.4, color="#7A3E00")
        add_arrow(ax, (x + 12, 102), (x + 12, 96), lw=1.4, color="#7A3E00")

    add_block(ax, (x0 + 18, 66), 44, 9, title="Concat  ->  144 channels",
              facecolor="#FFE6C2", edgecolor=COLOR_HUB_EDGE, fontsize_title=12)
    for x in bx:
        add_arrow(ax, (x + 12, 82), (x0 + 40, 75), lw=1.4, color="#7A3E00")

    add_block(ax, (x0 + 22, 50), 36, 9,
              title="Conv 1x1  144 -> 112  +  BN  +  SiLU",
              facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=11)
    add_arrow(ax, (x0 + 40, 66), (x0 + 40, 59), lw=1.4, color="#7A3E00")

    add_block(ax, (x0 + 27, 34), 26, 9, title="Output  (112 x 14 x 14)",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=11)
    add_arrow(ax, (x0 + 40, 50), (x0 + 40, 43), lw=1.4, color="#7A3E00")

    ax.text(x0 + 40, 24, "BAM + Triplet + KAN (full):  99.06 / 99.21",
            ha="center", fontsize=11.5, color=INK)
    ax.text(x0 + 40, 18, "BAM + Triplet pair:  98.25 / 98.36   (role conflict)",
            ha="center", fontsize=11.5, color=FAIL_RED, fontweight="bold")


def draw_v2(ax, x0):
    ax.text(x0 + 40, 129, "AttentionHub-v2  —  Sequential Cascade  (proposed)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=PROPOSED_EDGE)

    add_block(ax, (x0 + 27, 118), 26, 9, title="Input  (96 x 14 x 14)",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=11)

    add_block(ax, (x0 + 20, 102), 40, 9,
              title="Conv 1x1   96 -> 96  (reduce)",
              facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=11)

    add_block(ax, (x0 + 16, 84), 48, 12, title="Triplet Attention",
              sub="cross-dimensional spatial gating",
              facecolor=COLOR_ATTN_FILL, edgecolor=COLOR_ATTN_EDGE,
              fontsize_title=13, fontsize_sub=10.5)

    add_block(ax, (x0 + 16, 66), 48, 12, title="Squeeze-Excitation (SE)",
              sub="pure channel gating",
              facecolor=COLOR_ATTN_FILL, edgecolor=COLOR_ATTN_EDGE,
              fontsize_title=13, fontsize_sub=10.5)

    add_block(ax, (x0 + 20, 50), 40, 9,
              title="Conv 1x1   96 -> 112  (expand)",
              facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=11)

    add_block(ax, (x0 + 27, 34), 26, 9, title="Output  (112 x 14 x 14)",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=11)

    cx = x0 + 40
    for y1, y2 in [(118, 111), (102, 96), (84, 78), (66, 59), (50, 43)]:
        add_arrow(ax, (cx, y1), (cx, y2), lw=1.6, color=PROPOSED_EDGE)

    # Result panel
    ax.text(x0 + 40, 24, "Triplet → SE:   99.06 / 99.51",
            ha="center", fontsize=13, color=PROPOSED_EDGE, fontweight="bold")
    ax.text(x0 + 40, 18, "best subtype accuracy in the paper",
            ha="center", fontsize=11, color="#2F6B3A", style="italic")


def main():
    apply_rc()
    fig, ax = setup_canvas(figsize=(20, 12), xlim=(0, 200), ylim=(10, 142))

    ax.text(100, 139,
            "Hub v1 (parallel fusion)  vs  Hub v2 (sequential cascade)",
            ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(100, 134.5,
            "Same building blocks, different composition, different result.",
            ha="center", fontsize=11, color=MUTED, style="italic")

    draw_v1(ax, x0=4)
    draw_v2(ax, x0=110)

    # Divider
    ax.plot([100, 100], [12, 130], "--", color="#888888", lw=1.0, alpha=0.6)

    save_fig(fig, "01_architecture", "fig01c_attentionhub_v1_vs_v2", tight=False)


if __name__ == "__main__":
    main()
