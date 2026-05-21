"""fig01d — Internal block diagrams for Triplet, SE, BAM, KAN (clean redesign)."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _lib.style import apply_rc, save_fig
from _lib.layout import add_block, add_arrow


def _setup(ax, title, color):
    ax.set_xlim(0, 100); ax.set_ylim(0, 110); ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)


def panel_triplet(ax):
    """3 branches drawn in 3 separate vertical mini-flows — no horizontal cramming."""
    _setup(ax, "Triplet Attention", "#7A3E00")
    add_block(ax, (28, 95), 44, 8, title="Input  (C x H x W)",
              facecolor="#ECECEC", edgecolor="#5C5C5C", fontsize_title=11)

    # Three vertical branches
    bx = [4, 36, 68]
    for x, lbl in zip(bx, ["C-H permute", "C-W permute", "H-W (no permute)"]):
        add_block(ax, (x, 78), 28, 8, title=lbl, fontsize_title=10,
                  facecolor="#FFE1C2", edgecolor="#7A3E00")
        add_block(ax, (x, 60), 28, 13,
                  title="ZPool (max,mean)",
                  sub="Conv 7x7  -  BN  -  Sigmoid",
                  fontsize_title=10, fontsize_sub=9,
                  facecolor="#FFC78F", edgecolor="#7A3E00")
        # arrow from input to top of branch
        add_arrow(ax, (50, 95), (x + 14, 86), lw=1.2, color="#7A3E00")
        # arrow within branch
        add_arrow(ax, (x + 14, 78), (x + 14, 73), lw=1.4, color="#7A3E00")

    add_block(ax, (18, 38), 64, 10,
              title="Average gate  x  Input",
              fontsize_title=12,
              facecolor="#FFE6C2", edgecolor="#B36B00")
    for x in bx:
        add_arrow(ax, (x + 14, 60), (50, 48), lw=1.2, color="#7A3E00")

    add_block(ax, (28, 20), 44, 8, title="Output  (C x H x W)",
              facecolor="#ECECEC", edgecolor="#5C5C5C", fontsize_title=11)
    add_arrow(ax, (50, 38), (50, 28), lw=1.5, color="#7A3E00")


def panel_se(ax):
    _setup(ax, "Squeeze-Excitation (SE)", "#7A3E00")
    blocks = [
        (96, "Input  (C x H x W)", "#ECECEC", "#5C5C5C"),
        (84, "GAP  ->  (C x 1 x 1)", "#FFE1C2", "#7A3E00"),
        (72, "Conv 1x1  C -> C/16  +  SiLU", "#FFC78F", "#7A3E00"),
        (60, "Conv 1x1  C/16 -> C  +  Sigmoid", "#FFC78F", "#7A3E00"),
        (48, "Element-wise multiply with input", "#FFE6C2", "#B36B00"),
        (32, "Output  (C x H x W)", "#ECECEC", "#5C5C5C"),
    ]
    for y, lbl, fc, ec in blocks:
        add_block(ax, (10, y), 80, 8, title=lbl, fontsize_title=11,
                  facecolor=fc, edgecolor=ec)
    for y1, y2 in [(96, 92), (84, 80), (72, 68), (60, 56), (48, 40)]:
        add_arrow(ax, (50, y1), (50, y2), lw=1.5, color="#7A3E00")


def panel_bam(ax):
    _setup(ax, "BAM (Bottleneck Attention)", "#2C5C9C")
    add_block(ax, (28, 95), 44, 8, title="Input  (C x H x W)",
              facecolor="#ECECEC", edgecolor="#5C5C5C", fontsize_title=11)

    add_block(ax, (4, 64), 44, 18,
              title="Channel Gate",
              sub="GAP  ->  FC bottleneck\n->  expand",
              fontsize_title=12, fontsize_sub=10,
              facecolor="#DCE7F4", edgecolor="#2C5C9C")
    add_block(ax, (52, 64), 44, 18,
              title="Spatial Gate",
              sub="Conv 1x1  ->  3x3 dilated (d=4)\n->  Conv 1x1",
              fontsize_title=12, fontsize_sub=10,
              facecolor="#DCE7F4", edgecolor="#2C5C9C")
    add_arrow(ax, (40, 95), (26, 82), lw=1.4, color="#2C5C9C")
    add_arrow(ax, (60, 95), (74, 82), lw=1.4, color="#2C5C9C")

    add_block(ax, (18, 36), 64, 16,
              title="sigmoid(channel + spatial)",
              sub="x  Input  (residual multiply)",
              facecolor="#C8DCEF", edgecolor="#2C5C9C",
              fontsize_title=12, fontsize_sub=10)
    add_arrow(ax, (26, 64), (35, 52), lw=1.2, color="#2C5C9C")
    add_arrow(ax, (74, 64), (65, 52), lw=1.2, color="#2C5C9C")

    add_block(ax, (28, 16), 44, 8, title="Output  (C x H x W)",
              facecolor="#ECECEC", edgecolor="#5C5C5C", fontsize_title=11)
    add_arrow(ax, (50, 36), (50, 24), lw=1.5, color="#2C5C9C")


def panel_kan(ax):
    _setup(ax, "KAN-style Channel Attention", "#5B2C82")
    blocks = [
        (96, "Input  (C x H x W)", "#ECECEC", "#5C5C5C"),
        (84, "GAP  ->  (C,)", "#E7DCF4", "#5B2C82"),
        (70, "B-spline activation  (5 Gaussian bases)", "#D5C3EE", "#5B2C82"),
        (56, "+ SiLU residual   ->   Sigmoid", "#D5C3EE", "#5B2C82"),
        (42, "Multiply with input  (broadcast)", "#E7DCF4", "#5B2C82"),
        (28, "Output  (C x H x W)", "#ECECEC", "#5C5C5C"),
    ]
    for y, lbl, fc, ec in blocks:
        add_block(ax, (4, y), 92, 8, title=lbl, fontsize_title=11,
                  facecolor=fc, edgecolor=ec)
    for y1, y2 in [(96, 92), (84, 78), (70, 64), (56, 50), (42, 36)]:
        add_arrow(ax, (50, y1), (50, y2), lw=1.5, color="#5B2C82")


def main():
    apply_rc()
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    fig.suptitle("Attention Module Internals",
                 fontsize=17, fontweight="bold", y=0.998)
    panel_triplet(axes[0, 0])
    panel_se(axes[0, 1])
    panel_bam(axes[1, 0])
    panel_kan(axes[1, 1])
    fig.subplots_adjust(hspace=0.18, wspace=0.10, top=0.95)
    save_fig(fig, "01_architecture", "fig01d_attention_modules_internals", tight=False)


if __name__ == "__main__":
    main()
