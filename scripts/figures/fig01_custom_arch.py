"""fig01 — Custom EfficientNet V2 main architecture diagram (clean redesign).

5-stage backbone with AttentionHub-v2 (Triplet -> SE cascade) at Stage 4,
GAP, and dual-head split into binary (2 classes) and subtype (7 classes).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _lib.style import (apply_rc, save_fig,
                        COLOR_STAGE_FILL, COLOR_STAGE_EDGE,
                        COLOR_HUB_FILL, COLOR_HUB_EDGE,
                        COLOR_ATTN_FILL, COLOR_ATTN_EDGE,
                        COLOR_HEAD_FILL, COLOR_HEAD_EDGE,
                        COLOR_IO_FILL, COLOR_IO_EDGE)
from _lib.layout import add_block, add_arrow, setup_canvas


def main():
    apply_rc()
    fig, ax = setup_canvas(figsize=(22, 11), xlim=(0, 220), ylim=(0, 110))

    # ---- Title ----
    ax.text(110, 105,
            "Custom EfficientNet V2 — Dual-Head Classifier with AttentionHub-v2 (Triplet then SE Cascade)",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#1B2631")
    ax.text(110, 99,
            "4.79 M parameters   |   0.493 GFLOPs   |   99.06 / 99.51 binary / subtype accuracy",
            ha="center", va="center", fontsize=12, color="#555555", style="italic")

    # ---- Main horizontal pipeline ----
    y_main = 60
    h_main = 18

    blocks = [
        (2, "Input Image", "RGB", "3 x 224 x 224", COLOR_IO_FILL, COLOR_IO_EDGE),
        (22, "Stem", "Conv 3x3 + BN + SiLU\nstride 2", "16 x 112 x 112", COLOR_STAGE_FILL, COLOR_STAGE_EDGE),
        (44, "Stage 1", "FusedMBConv x2\nstride 2", "32 x 56 x 56", COLOR_STAGE_FILL, COLOR_STAGE_EDGE),
        (66, "Stage 2", "FusedMBConv\nstride 2", "48 x 28 x 28", COLOR_STAGE_FILL, COLOR_STAGE_EDGE),
        (88, "Stage 3", "MBConv\nstride 2", "96 x 14 x 14", COLOR_STAGE_FILL, COLOR_STAGE_EDGE),
        # Stage 4 (hub) goes here, drawn separately
        (152, "Stage 5", "MBConv + SE\nstride 2", "192 x 7 x 7", COLOR_STAGE_FILL, COLOR_STAGE_EDGE),
        (174, "GAP", "AdaptiveAvgPool2d(1)", "192", COLOR_IO_FILL, COLOR_IO_EDGE),
    ]
    w_main = 18

    block_rects = {}
    for x, title, sub, shape, fc, ec in blocks:
        add_block(ax, (x, y_main), w_main, h_main, title, sub, shape,
                  facecolor=fc, edgecolor=ec,
                  fontsize_title=11.5, fontsize_sub=9.5, fontsize_shape=8.5)
        block_rects[title] = (x, y_main, w_main, h_main)

    # ---- Stage 4: AttentionHub-v2 (enlarged) ----
    hub_x, hub_y = 110, 28
    hub_w, hub_h = 36, 56

    hub_box = mpatches.FancyBboxPatch(
        (hub_x + 0.2, hub_y + 0.2), hub_w - 0.4, hub_h - 0.4,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        facecolor=COLOR_HUB_FILL, edgecolor=COLOR_HUB_EDGE, linewidth=2.4)
    ax.add_patch(hub_box)

    ax.text(hub_x + hub_w / 2, hub_y + hub_h - 4,
            "Stage 4 - AttentionHub-v2",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#7A3E00")
    ax.text(hub_x + hub_w / 2, hub_y + hub_h - 7.5,
            "(Triplet  then  SE  sequential cascade)",
            ha="center", va="center", fontsize=10, color="#7A3E00", style="italic")
    ax.text(hub_x + hub_w / 2, hub_y - 3, "Shape:  112 x 14 x 14",
            ha="center", va="top", fontsize=9.5, color="#555555",
            style="italic", family="monospace")

    # Inner cascade (4 stacked blocks)
    inner_w = hub_w - 5
    inner_x = hub_x + 2.5
    inner_h = 8.5
    gap = 2.0
    top_y = hub_y + hub_h - 13   # below the title
    inner_ys = [top_y - i * (inner_h + gap) for i in range(4)]
    inner_specs = [
        ("Conv 1x1  (96 -> 96)", "BN + SiLU  -  reduce", "#FFF4E0", COLOR_HUB_EDGE),
        ("Triplet Attention", "3 cross-dim spatial branches", COLOR_ATTN_FILL, COLOR_ATTN_EDGE),
        ("Squeeze-Excitation (SE)", "GAP -> FC -> sigmoid gate", COLOR_ATTN_FILL, COLOR_ATTN_EDGE),
        ("Conv 1x1  (96 -> 112)", "BN + SiLU  -  expand", "#FFF4E0", COLOR_HUB_EDGE),
    ]
    for (t, s, fc, ec), yi in zip(inner_specs, inner_ys):
        add_block(ax, (inner_x, yi - inner_h), inner_w, inner_h,
                  title=t, sub=s,
                  facecolor=fc, edgecolor=ec, lw=1.4,
                  fontsize_title=10.5, fontsize_sub=9, rounding=0.4)

    # Inner arrows (top down)
    for i in range(3):
        y1 = inner_ys[i] - inner_h
        y2 = inner_ys[i + 1]
        add_arrow(ax, (hub_x + hub_w / 2, y1), (hub_x + hub_w / 2, y2),
                  color=COLOR_HUB_EDGE, lw=1.6)

    # Backbone arrows (input -> stem -> S1 -> S2 -> S3 -> hub -> S5 -> GAP)
    arrow_y = y_main + h_main / 2
    backbone_xs = [
        (2 + w_main, 22),                # Input -> Stem
        (22 + w_main, 44),               # Stem -> S1
        (44 + w_main, 66),               # S1 -> S2
        (66 + w_main, 88),               # S2 -> S3
        (88 + w_main, hub_x),            # S3 -> Hub
        (hub_x + hub_w, 152),            # Hub -> S5
        (152 + w_main, 174),             # S5 -> GAP
    ]
    for x1, x2 in backbone_xs:
        add_arrow(ax, (x1, arrow_y), (x2, arrow_y), lw=2.0)

    # Dropout marker between GAP and fork
    ax.text(174 + w_main + 4.5, arrow_y + 5, "Dropout(p)",
            ha="center", va="center", fontsize=10,
            color="#1B2631", style="italic")
    fork_x = 174 + w_main + 9
    add_arrow(ax, (174 + w_main, arrow_y), (fork_x, arrow_y), lw=2.0)
    ax.plot([fork_x], [arrow_y], "o", color="#1B2631", markersize=8, zorder=10)

    # ---- Two heads ----
    head_x = fork_x + 4
    head_w = 32
    head_h = 12
    bin_y_center = arrow_y + 14
    sub_y_center = arrow_y - 14

    add_block(ax, (head_x, bin_y_center - head_h / 2), head_w, head_h,
              title="Binary Head",
              sub="Linear 192 -> 512  |  ReLU  |  Dropout  |  Linear 512 -> 2",
              shape="Benign  /  Malignant",
              facecolor=COLOR_HEAD_FILL, edgecolor=COLOR_HEAD_EDGE,
              fontsize_title=12, fontsize_sub=9.5, fontsize_shape=10)

    add_block(ax, (head_x, sub_y_center - head_h / 2), head_w, head_h,
              title="Subtype Head",
              sub="Linear 192 -> 512  |  ReLU  |  Dropout  |  Linear 512 -> 7",
              shape="CaS / CoS / Gum / MC / OC / OLP / OT",
              facecolor=COLOR_HEAD_FILL, edgecolor=COLOR_HEAD_EDGE,
              fontsize_title=12, fontsize_sub=9.5, fontsize_shape=10)

    # Fork to head arrows
    add_arrow(ax, (fork_x, arrow_y), (head_x, bin_y_center), lw=2.0)
    add_arrow(ax, (fork_x, arrow_y), (head_x, sub_y_center), lw=2.0)

    # ---- Legend ----
    leg_y = 8
    leg_h = 4
    leg_w = 4
    leg_items = [
        (COLOR_STAGE_FILL, COLOR_STAGE_EDGE, "Backbone stage"),
        (COLOR_HUB_FILL, COLOR_HUB_EDGE, "AttentionHub-v2 (proposed)"),
        (COLOR_ATTN_FILL, COLOR_ATTN_EDGE, "Attention module"),
        (COLOR_HEAD_FILL, COLOR_HEAD_EDGE, "Classification head"),
        (COLOR_IO_FILL, COLOR_IO_EDGE, "Input / pooling"),
    ]
    lx = 8
    for fc, ec, lbl in leg_items:
        rect = mpatches.FancyBboxPatch((lx, leg_y), leg_w, leg_h,
                                       boxstyle="round,pad=0.02,rounding_size=0.3",
                                       facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(lx + leg_w + 1.5, leg_y + leg_h / 2, lbl,
                ha="left", va="center", fontsize=10.5)
        lx += 42

    save_fig(fig, "01_architecture", "fig01_custom_efficientnet_v2_arch", tight=False)


if __name__ == "__main__":
    main()
