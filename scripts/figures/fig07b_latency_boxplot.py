"""fig07b — Latency boxplot reconstructed from min/p50/p95/p99/max + mean/std."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        PROPOSED_COLOR, PROPOSED_EDGE, BASELINE_COLOR,
                        BASELINE_EDGE, MUTED)
from _lib.data_loader import collect_table1, PROPOSED_V2


def main():
    apply_rc()
    rows = sorted(collect_table1(), key=lambda r: r["p50_ms"])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(11.6, 6.7))
    y = np.arange(n)

    for i, r in enumerate(rows):
        is_p = r["run"] == PROPOSED_V2
        fc = PROPOSED_COLOR if is_p else BASELINE_COLOR
        ec = PROPOSED_EDGE if is_p else BASELINE_EDGE
        # whisker: min .. max
        ax.plot([r["lat_min_ms"], r["lat_max_ms"]], [i, i], color=ec, lw=1.0,
                alpha=0.55, zorder=2)
        for xw in (r["lat_min_ms"], r["lat_max_ms"]):
            ax.plot([xw, xw], [i - 0.13, i + 0.13], color=ec, lw=1.0,
                    alpha=0.55, zorder=2)
        # box: q1 .. q3 proxy
        sig = max(r["lat_std_ms"], 0.05)
        q1 = max(r["lat_min_ms"], r["p50_ms"] - 0.67 * sig)
        q3 = min(r["lat_max_ms"], r["p95_ms"])
        ax.add_patch(plt.Rectangle((q1, i - 0.27), q3 - q1, 0.54, facecolor=fc,
                                   edgecolor=ec, lw=1.0,
                                   alpha=0.92 if is_p else 0.80, zorder=3))
        # median tick
        ax.plot([r["p50_ms"], r["p50_ms"]], [i - 0.27, i + 0.27], color="white",
                lw=2.2, zorder=4)
        # P95 / P99 markers
        ax.plot(r["p95_ms"], i, "v", color=ec, ms=6, zorder=4)
        ax.plot(r["p99_ms"], i, "x", color=ec, ms=6, mew=1.5, zorder=4)

    # aligned value column on the right
    x_lo = min(r["lat_min_ms"] for r in rows) * 0.85
    x_data_hi = max(r["lat_max_ms"] for r in rows)
    label_x = x_data_hi * 1.05
    for i, r in enumerate(rows):
        is_p = r["run"] == PROPOSED_V2
        ax.text(label_x, i, f"P50 {r['p50_ms']:.1f} ms", va="center", ha="left",
                fontsize=8.3, color=PROPOSED_EDGE if is_p else MUTED,
                fontweight="bold" if is_p else "normal")
    ax.set_xlim(x_lo, label_x * 1.16)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] + ("  (proposed)" if r["run"] == PROPOSED_V2 else "")
                        for r in rows])
    for lbl, r in zip(ax.get_yticklabels(), rows):
        if r["run"] == PROPOSED_V2:
            lbl.set_color(PROPOSED_EDGE)
            lbl.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Latency per image (ms)")
    ax.set_title("Per-model inference latency distributions")
    style_axes(ax, grid="x")
    ax.tick_params(axis="y", length=0)

    handles = [
        mlines.Line2D([], [], color="#888888", lw=2.2, label="median (P50)"),
        mlines.Line2D([], [], color="#888888", marker="v", ls="none", ms=6,
                      label="P95"),
        mlines.Line2D([], [], color="#888888", marker="x", ls="none", ms=6,
                      mew=1.5, label="P99"),
    ]
    legend_clean(ax, handles=handles, loc="lower right", ncol=3)
    save_fig(fig, "07_latency", "fig07b_latency_boxplot")


if __name__ == "__main__":
    main()
