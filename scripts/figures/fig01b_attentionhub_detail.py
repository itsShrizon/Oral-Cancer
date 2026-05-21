"""fig01b — Zoomed AttentionHub-v2 with full Triplet + SE internals (clean)."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.patches as mpatches
from _lib.style import (apply_rc, save_fig, COLOR_HUB_FILL, COLOR_HUB_EDGE,
                        COLOR_ATTN_FILL, COLOR_ATTN_EDGE, COLOR_IO_FILL,
                        COLOR_IO_EDGE, COLOR_HEAD_EDGE, INK, MUTED)
from _lib.layout import add_block, add_arrow, setup_canvas


def main():
    apply_rc()
    fig, ax = setup_canvas(figsize=(18, 12), xlim=(0, 200), ylim=(0, 140))

    ax.text(100, 134, "AttentionHub-v2 — Triplet then SE Sequential Cascade",
            ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(100, 128.5,
            "Channels 96  →  cross-dim spatial gating (Triplet)  →  pure-channel gating (SE)  →  112",
            ha="center", va="center", fontsize=11.5, color=MUTED, style="italic")

    # --- Left pane: vertical cascade ---
    pane_left = 6
    box_w = 80
    box_h = 9

    # Input
    add_block(ax, (pane_left, 110), box_w, box_h,
              title="Input feature map  (from Stage 3)",
              shape="96 x 14 x 14",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=12)

    # Reduce
    add_block(ax, (pane_left, 96), box_w, box_h,
              title="Conv 1x1  (96 -> 96)  +  BN  +  SiLU",
              sub="reduce",
              facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=12)

    # Triplet outer
    triplet_y, triplet_h = 60, 28
    triplet_box = mpatches.FancyBboxPatch(
        (pane_left + 0.2, triplet_y + 0.2), box_w - 0.4, triplet_h - 0.4,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        facecolor=COLOR_ATTN_FILL, edgecolor=COLOR_ATTN_EDGE, linewidth=2.0)
    ax.add_patch(triplet_box)
    ax.text(pane_left + box_w / 2, triplet_y + triplet_h - 3.5,
            "Triplet Attention",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#5C2E00")
    ax.text(pane_left + box_w / 2, triplet_y + triplet_h - 7,
            "3 cross-dimensional branches  (averaged)",
            ha="center", va="center", fontsize=10, color="#5C2E00", style="italic")

    sub_w = (box_w - 12) / 3
    sub_h = 12
    sub_y = triplet_y + 4
    for i, lbl in enumerate(["C-H permute", "C-W permute", "H-W (no permute)"]):
        sub_x = pane_left + 3 + i * (sub_w + 1.5)
        add_block(ax, (sub_x, sub_y), sub_w, sub_h,
                  title=lbl,
                  sub="ZPool  ->  Conv 7x7\nBN  ->  Sigmoid",
                  facecolor="#FFE1C2", edgecolor=COLOR_ATTN_EDGE,
                  fontsize_title=10, fontsize_sub=8.5, rounding=0.4)

    # SE
    add_block(ax, (pane_left, 40), box_w, 14,
              title="Squeeze-Excitation (SE)",
              sub="GAP  ->  Conv1x1 96 -> 6  +  SiLU  ->  Conv1x1 6 -> 96  +  Sigmoid  ->  multiply",
              facecolor=COLOR_ATTN_FILL, edgecolor=COLOR_ATTN_EDGE,
              fontsize_title=13, fontsize_sub=9.5)

    # Expand
    add_block(ax, (pane_left, 26), box_w, 9,
              title="Conv 1x1  (96 -> 112)  +  BN  +  SiLU",
              sub="expand",
              facecolor="#FFF4E0", edgecolor=COLOR_HUB_EDGE, fontsize_title=12)

    # Output
    add_block(ax, (pane_left, 12), box_w, 9,
              title="Output feature map  (to Stage 5)",
              shape="112 x 14 x 14",
              facecolor=COLOR_IO_FILL, edgecolor=COLOR_IO_EDGE, fontsize_title=12)

    # Vertical arrows
    cx = pane_left + box_w / 2
    for y1, y2 in [(110, 105), (96, 88), (60, 54), (40, 35), (26, 21)]:
        add_arrow(ax, (cx, y1), (cx, y2), color=COLOR_HUB_EDGE, lw=1.8)

    # --- Right pane: design rationale & result ---
    rx = 104
    ax.text(rx, 122, "Design rationale",
            ha="left", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.plot([rx, rx + 80], [118.2, 118.2], color=COLOR_ATTN_EDGE, lw=1.2)

    bullets = [
        "Triplet performs cross-dimensional spatial gating\nacross the (H, W) and mixed axes.",
        "SE performs pure per-channel re-weighting, with\nno spatial role overlapping Triplet.",
        "Disjoint roles avoid the 98.36 % accuracy ceiling\nseen in BAM+Triplet and v2-EMA pairings.",
        "The sequential cascade preserves information —\nno channel concat, no LayerScale.",
    ]
    y = 112
    for b in bullets:
        nlines = b.count("\n") + 1
        ax.text(rx + 1.5, y, "•", ha="left", va="top", fontsize=13,
                color=COLOR_ATTN_EDGE, fontweight="bold")
        ax.text(rx + 6, y, b, ha="left", va="top", fontsize=11,
                color=INK, linespacing=1.55)
        y -= 4.2 + nlines * 5.0

    # --- Result card ---
    card_x, card_y, card_w, card_h = rx, 20, 80, 39
    card = mpatches.FancyBboxPatch(
        (card_x, card_y), card_w, card_h,
        boxstyle="round,pad=0.4,rounding_size=0.6",
        facecolor="#EAF4EA", edgecolor=COLOR_HEAD_EDGE, linewidth=1.4)
    ax.add_patch(card)
    cx = card_x + card_w / 2
    ax.text(cx, card_y + card_h - 6,
            "Result  —  best subtype accuracy in the paper",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="#2F6B3A")
    ax.text(cx, card_y + card_h - 17.5, "99.51 %",
            ha="center", va="center", fontsize=25, fontweight="bold",
            color="#2F6B3A")
    ax.text(cx, card_y + card_h - 26,
            "subtype accuracy        99.06 % binary accuracy",
            ha="center", va="center", fontsize=10.2, color="#3C6B44")
    ax.plot([card_x + 9, card_x + card_w - 9], [card_y + 9.5, card_y + 9.5],
            color="#C2DCC2", lw=1.0)
    ax.text(cx, card_y + 5.6,
            "4.79 M params      0.493 GFLOPs      52.3 MB GPU      9.86 ms P50",
            ha="center", va="center", fontsize=9.6, color="#3C6B44")

    save_fig(fig, "01_architecture", "fig01b_attentionhub_v2_detail", tight=False)


if __name__ == "__main__":
    main()
