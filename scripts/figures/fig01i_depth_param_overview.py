"""fig01i — Side-by-side depth and parameter comparison of 5 architectures."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from _lib.style import (apply_rc, save_fig, style_axes,
                        PROPOSED_COLOR, PROPOSED_EDGE,
                        BASELINE_COLOR, BASELINE_EDGE, INK, MUTED)


def main():
    apply_rc()
    models = ["ResNet50", "Inception V3", "Swin-T", "EffNetV2-B2", "Custom V2 (Hub v2)"]
    params_m = [25.61, 23.85, 28.29, 10.00, 4.79]
    gflops = [4.134, 2.838, 4.372, 1.100, 0.493]
    # Approximate depth (number of trainable blocks / layers) for a coarse comparison
    depth = [50, 48, 28, 31, 22]

    proposed = [m.startswith("Custom V2") for m in models]
    fills = [PROPOSED_COLOR if p else BASELINE_COLOR for p in proposed]
    edges = [PROPOSED_EDGE if p else BASELINE_EDGE for p in proposed]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    x = np.arange(len(models))

    panels = [
        (axes[0], params_m, "Parameters (M)", "Parameter count", "{:.1f}", 0.18),
        (axes[1], gflops, "GFLOPs", "Inference compute", "{:.2f}", 0.18),
        (axes[2], depth, "≈ block / layer count", "Architectural depth (approx.)", "{:.0f}", 0.18),
    ]
    for ax, vals, ylab, title, fmt, headroom in panels:
        bars = ax.bar(x, vals, width=0.66, color=fills, edgecolor=edges, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=22, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * (1 + headroom))
        for b, v, p in zip(bars, vals, proposed):
            ax.annotate(fmt.format(v),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.8,
                        color=PROPOSED_EDGE if p else MUTED,
                        fontweight="bold" if p else "normal")
        style_axes(ax, grid="y")
        for lbl, p in zip(ax.get_xticklabels(), proposed):
            if p:
                lbl.set_color(PROPOSED_EDGE)
                lbl.set_fontweight("bold")

    fig.suptitle("Architecture footprint  —  proposed model vs four baseline backbones",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "01_architecture", "fig01i_depth_param_overview", tight=False)


if __name__ == "__main__":
    main()
