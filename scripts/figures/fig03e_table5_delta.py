"""fig03e — Paired bars vs strongest baselines (Table 5 deltas)."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, panel_tag,
                        PROPOSED_COLOR, PROPOSED_EDGE, BASELINE_COLOR,
                        BASELINE_EDGE, GOOD_GREEN, INK, MUTED)
from _lib.data_loader import collect_table1, PROPOSED_V2


def main():
    apply_rc()
    rows = {r["run"]: r for r in collect_table1()}
    a = rows["efficientnet_v2b2"]
    b = rows["inception_v3"]
    p = rows[PROPOSED_V2]
    series = [a, b, p]
    is_prop = [False, False, True]
    short = ["EffNetV2-B2", "Inception V3", "Custom V2 (Hub v2)"]

    metrics = [
        ("Binary accuracy (%)", lambda r: r["binary_acc"] * 100, True, "{:.2f}", True),
        ("Subtype accuracy (%)", lambda r: r["subtype_acc"] * 100, True, "{:.2f}", True),
        ("Parameters (M) — lower better", lambda r: r["params_m"], False, "{:.2f}", False),
        ("GFLOPs — lower better", lambda r: r["gflops"], False, "{:.2f}", False),
        ("Model size (MB) — lower better", lambda r: r["size_mb"], False, "{:.1f}", False),
        ("GPU peak (MB) — lower better", lambda r: r["gpu_peak_mb"], False, "{:.1f}", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    axes = axes.flatten()
    fills = [PROPOSED_COLOR if pr else BASELINE_COLOR for pr in is_prop]
    edges = [PROPOSED_EDGE if pr else BASELINE_EDGE for pr in is_prop]
    tags = "abcdef"

    for k, (ax, (name, getter, higher_better, fmt, zoom)) in enumerate(zip(axes, metrics)):
        vals = [getter(r) for r in series]
        x = np.arange(3)
        bars = ax.bar(x, vals, width=0.62, color=fills, edgecolor=edges, lw=0.8)
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=16, ha="right")
        if zoom:
            ax.set_ylim(min(vals) - 1.6, 100.9)
        else:
            ax.set_ylim(0, max(vals) * 1.24)
        for bar, v, pr in zip(bars, vals, is_prop):
            ax.annotate(fmt.format(v),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8.4,
                        color=PROPOSED_EDGE if pr else MUTED,
                        fontweight="bold" if pr else "normal")
        best_i = int(np.argmax(vals) if higher_better else np.argmin(vals))
        ax.annotate("best", (best_i, vals[best_i]), xytext=(0, 20),
                    textcoords="offset points", ha="center", fontsize=8.2,
                    color=GOOD_GREEN, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#E8F5EE",
                              edgecolor=GOOD_GREEN, linewidth=0.8))
        style_axes(ax, grid="y")
        for lbl, pr in zip(ax.get_xticklabels(), is_prop):
            if pr:
                lbl.set_color(PROPOSED_EDGE)
                lbl.set_fontweight("bold")
        panel_tag(ax, tags[k])

    fig.suptitle("Proposed Custom V2 (Hub v2) vs. the two strongest baselines  (Table 5)",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "03_benchmark", "fig03e_table5_delta_bars", tight=False)


if __name__ == "__main__":
    main()
