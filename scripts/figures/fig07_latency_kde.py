"""fig07 — Inference-latency ridgeline (one density curve per model).

Replaces the old 11-curve overlay: a ridgeline keeps every model legible.
Densities are Gaussian approximations from each model's latency mean / std.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, PROPOSED_COLOR, PROPOSED_EDGE,
                        BASELINE_COLOR, BASELINE_EDGE, MUTED, HAIRLINE)
from _lib.data_loader import collect_table1, PROPOSED_V2

RIDGE_HEIGHT = 1.95   # how far each ridge rises, in row units (controls overlap)


def main():
    apply_rc()
    rows = [r for r in collect_table1() if r["lat_mean_ms"] > 0]
    rows.sort(key=lambda r: r["lat_mean_ms"], reverse=True)  # fastest ends on top
    n = len(rows)

    xmin = min(r["lat_min_ms"] for r in rows)
    xmax = max(r["lat_max_ms"] for r in rows)
    x = np.linspace(max(0.0, xmin * 0.6), xmax * 1.04, 600)

    fig, ax = plt.subplots(figsize=(10.6, 7.4))

    for i, r in enumerate(rows):
        mu = r["lat_mean_ms"]
        sig = max(r["lat_std_ms"], 0.12)
        pdf = np.exp(-0.5 * ((x - mu) / sig) ** 2)
        pdf = pdf / pdf.max()
        curve = i + pdf * RIDGE_HEIGHT
        is_p = r["run"] == PROPOSED_V2
        fc = PROPOSED_COLOR if is_p else BASELINE_COLOR
        ec = PROPOSED_EDGE if is_p else BASELINE_EDGE
        z = (1000 if is_p else 10) + (n - i)
        ax.plot([x[0], x[-1]], [i, i], color=HAIRLINE, lw=0.6, zorder=z - 1)
        ax.fill_between(x, i, curve, color=fc, alpha=0.88 if is_p else 0.80,
                        lw=0, zorder=z)
        ax.plot(x, curve, color=ec, lw=1.6 if is_p else 0.9, zorder=z + 0.4)
        p50 = r["p50_ms"]
        yk = i + float(np.interp(p50, x, pdf)) * RIDGE_HEIGHT
        ax.plot([p50, p50], [i, yk], color=ec, lw=1.1, ls=":", zorder=z + 0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] + ("  (proposed)" if r["run"] == PROPOSED_V2 else "")
                        for r in rows])
    for lbl, r in zip(ax.get_yticklabels(), rows):
        if r["run"] == PROPOSED_V2:
            lbl.set_color(PROPOSED_EDGE)
            lbl.set_fontweight("bold")

    ax.set_ylim(-0.6, n - 1 + RIDGE_HEIGHT + 0.35)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Inference latency per image (ms)")
    ax.set_title("Inference latency distributions — per-model density ridgeline  (lower is better)")

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=MUTED, labelcolor="#444444", length=3.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E6E6", lw=0.7, ls=":")
    ax.set_axisbelow(True)

    ax.plot([], [], color=MUTED, ls=":", lw=1.2, label="median (P50)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    save_fig(fig, "07_latency", "fig07_latency_kde_overlay")


if __name__ == "__main__":
    main()
