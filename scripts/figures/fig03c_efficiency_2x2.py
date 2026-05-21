"""fig03c — 2x2 efficiency comparison: GPU peak, P50/P95 latency, training time."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, panel_tag,
                        PROPOSED_COLOR, PROPOSED_EDGE, BASELINE_COLOR,
                        BASELINE_EDGE, GOOD_GREEN, INK, MUTED)
from _lib.data_loader import collect_table1, PROPOSED_V2


def _bar(ax, rows, key, title, ylabel, fmt="{:.1f}", tag="a"):
    vals = [r.get(key, 0) for r in rows]
    labels = [r["label"] for r in rows]
    is_prop = [r["run"] == PROPOSED_V2 for r in rows]
    fills = [PROPOSED_COLOR if p else BASELINE_COLOR for p in is_prop]
    edges = [PROPOSED_EDGE if p else BASELINE_EDGE for p in is_prop]
    x = np.arange(len(rows))
    bars = ax.bar(x, vals, width=0.68, color=fills, edgecolor=edges, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.22)

    for b, v, p in zip(bars, vals, is_prop):
        if v > 0:
            ax.annotate(fmt.format(v),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.8,
                        color=PROPOSED_EDGE if p else MUTED,
                        fontweight="bold" if p else "normal")

    # best = lowest non-zero value
    pos = [(i, v) for i, v in enumerate(vals) if v > 0]
    best_i = min(pos, key=lambda t: t[1])[0]
    ax.annotate("best", (best_i, vals[best_i]), xytext=(0, 19),
                textcoords="offset points", ha="center", fontsize=8.0,
                color=GOOD_GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#E8F5EE",
                          edgecolor=GOOD_GREEN, linewidth=0.8))

    style_axes(ax, grid="y")
    for lbl, p in zip(ax.get_xticklabels(), is_prop):
        if p:
            lbl.set_color(PROPOSED_EDGE)
            lbl.set_fontweight("bold")
    panel_tag(ax, tag)


def main():
    apply_rc()
    rows = collect_table1()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))
    _bar(axes[0, 0], rows, "gpu_peak_mb", "GPU peak memory  (lower is better)", "MB", tag="a")
    _bar(axes[0, 1], rows, "p50_ms", "P50 inference latency  (lower is better)", "ms", fmt="{:.2f}", tag="b")
    _bar(axes[1, 0], rows, "p95_ms", "P95 inference latency  (lower is better)", "ms", fmt="{:.2f}", tag="c")
    _bar(axes[1, 1], rows, "train_time_min", "Total training time  (lower is better)", "min", tag="d")
    fig.suptitle("Efficiency comparison  —  proposed Custom V2 (Hub v2) vs nine baselines",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "03_benchmark", "fig03c_efficiency_2x2", tight=False)


if __name__ == "__main__":
    main()
