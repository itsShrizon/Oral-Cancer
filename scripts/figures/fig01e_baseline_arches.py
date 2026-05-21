"""fig01e/f/g/h — Coarse architecture diagrams for ResNet50, EfficientNetV2-B2,
Inception V3, Swin-T.  All saved by a single script."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from _lib.style import (apply_rc, save_fig, COLOR_STAGE_FILL, COLOR_STAGE_EDGE,
                        COLOR_ATTN_FILL, COLOR_ATTN_EDGE, INK)
from _lib.layout import add_block, add_arrow, setup_canvas


def _pipeline(ax, blocks, total_w=110, y=18, h=7, title=None,
              palette=(COLOR_STAGE_FILL, COLOR_STAGE_EDGE), accent_idx=None):
    """blocks = [(name, sub, shape), ...]. Lays them out horizontally."""
    if title:
        ax.text(4, 30.5, title, ha="left", fontsize=13, fontweight="bold", color=INK)
    n = len(blocks)
    w = (total_w - 8) / n - 1.5
    x = 4
    centers = []
    for i, (name, sub, shape) in enumerate(blocks):
        fc, ec = palette
        if accent_idx is not None and i == accent_idx:
            fc, ec = COLOR_ATTN_FILL, COLOR_ATTN_EDGE
        add_block(ax, (x, y), w, h, title=name, sub=sub, shape=shape,
                  facecolor=fc, edgecolor=ec, fontsize_title=9.5, fontsize_sub=8)
        centers.append(x + w / 2)
        if i < n - 1:
            add_arrow(ax, (x + w, y + h / 2), (x + w + 1.5, y + h / 2), lw=1.4)
        x += w + 1.5
    return centers


def draw_resnet50():
    apply_rc()
    fig, ax = setup_canvas(figsize=(16.5, 3.3), xlim=(0, 112), ylim=(13.5, 33))
    _pipeline(ax,
              [("Input", "RGB", "3 × 224²"),
               ("Stem", "Conv7×7 s2 + MaxPool", "64 × 56²"),
               ("Stage 1", "Bottleneck ×3", "256 × 56²"),
               ("Stage 2", "Bottleneck ×4", "512 × 28²"),
               ("Stage 3", "Bottleneck ×6", "1024 × 14²"),
               ("Stage 4", "Bottleneck ×3", "2048 × 7²"),
               ("GAP", "—", "2048"),
               ("FC", "Linear 2048→N", "N")],
              total_w=110, y=18, h=8,
              title="ResNet50 — 25.61 M params • 4.13 GFLOPs")
    save_fig(fig, "01_architecture", "fig01e_resnet50_arch")


def draw_effnetv2b2():
    apply_rc()
    fig, ax = setup_canvas(figsize=(16.5, 3.3), xlim=(0, 112), ylim=(13.5, 33))
    _pipeline(ax,
              [("Input", "RGB", "3 × 224²"),
               ("Stem", "Conv3×3 s2", "32 × 112²"),
               ("Stage 1", "FusedMBConv ×2", "16 × 112²"),
               ("Stage 2", "FusedMBConv ×3", "32 × 56²"),
               ("Stage 3", "FusedMBConv ×3", "56 × 28²"),
               ("Stage 4", "MBConv ×4", "104 × 14²"),
               ("Stage 5", "MBConv ×6", "120 × 14²"),
               ("Stage 6", "MBConv ×10", "208 × 7²"),
               ("Head", "Conv + GAP + FC", "1408 → N")],
              total_w=110, y=18, h=8,
              title="EfficientNetV2-B2 — 10.00 M params • 1.10 GFLOPs")
    save_fig(fig, "01_architecture", "fig01f_efficientnetv2b2_arch")


def draw_inceptionv3():
    apply_rc()
    fig, ax = setup_canvas(figsize=(16.5, 3.3), xlim=(0, 112), ylim=(13.5, 33))
    _pipeline(ax,
              [("Input", "RGB", "3 × 299²"),
               ("Stem", "Conv stack s2 ×3", "192 × 35²"),
               ("Inception A ×3", "1×1 / 3×3 / 5×5 / pool", "288 × 35²"),
               ("Reduction A", "stride-2 grids", "768 × 17²"),
               ("Inception B ×4", "asym 1×7 / 7×1", "768 × 17²"),
               ("Reduction B", "stride-2 grids", "1280 × 8²"),
               ("Inception C ×2", "expanded mix", "2048 × 8²"),
               ("GAP", "—", "2048"),
               ("FC", "Linear 2048→N", "N")],
              total_w=110, y=18, h=8,
              title="Inception V3 — 23.85 M params • 2.84 GFLOPs")
    save_fig(fig, "01_architecture", "fig01g_inceptionv3_arch")


def draw_swint():
    apply_rc()
    fig, ax = setup_canvas(figsize=(16.5, 3.3), xlim=(0, 112), ylim=(13.5, 33))
    _pipeline(ax,
              [("Input", "RGB", "3 × 224²"),
               ("Patch Embed", "4×4 conv", "96 × 56²"),
               ("Stage 1", "Swin Block ×2 (W-MSA + SW-MSA)", "96 × 56²"),
               ("Patch Merge 1", "→ 2× channels", "192 × 28²"),
               ("Stage 2", "Swin Block ×2", "192 × 28²"),
               ("Patch Merge 2", "→ 2× channels", "384 × 14²"),
               ("Stage 3", "Swin Block ×6", "384 × 14²"),
               ("Patch Merge 3", "→ 2× channels", "768 × 7²"),
               ("Stage 4 + Head", "Block ×2 → LN → GAP → FC", "768 → N")],
              total_w=110, y=18, h=8,
              title="Swin-T — 28.29 M params • 4.37 GFLOPs",
              palette=("#E9E3F3", "#6B5B95"))
    save_fig(fig, "01_architecture", "fig01h_swint_arch")


def main():
    draw_resnet50()
    draw_effnetv2b2()
    draw_inceptionv3()
    draw_swint()


if __name__ == "__main__":
    main()
