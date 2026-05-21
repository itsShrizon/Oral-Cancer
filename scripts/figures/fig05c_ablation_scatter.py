"""fig05c — Ablation params vs accuracy scatter with leader-line callouts.

Ablation cells share an almost identical parameter count, so they form a tight
vertical cluster. Labels are placed in an evenly-spaced column on the right and
linked to their point with a thin leader line — collision-free by construction.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        PROPOSED_COLOR, PROPOSED_EDGE, BASELINE_COLOR,
                        BASELINE_EDGE, FAIL_RED, SUBTLE_INK, ABLATION_LABELS)
from _lib.data_loader import collect_ablation, collect_v2

CEILING = 98.36


def main():
    apply_rc()
    rows = collect_ablation()
    v2 = collect_v2()

    xs = [r["params_m"] for r in rows]
    ys = [r["subtype_acc"] * 100 for r in rows]
    labels = [ABLATION_LABELS.get(r["variant"], r["variant"]) for r in rows]
    if v2:
        xs.append(v2["params_m"])
        ys.append(v2["subtype_acc"] * 100)
        labels.append("Triplet → SE  (v2)")
    n = len(xs)
    pidx = n - 1 if v2 else -1

    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    fig.subplots_adjust(left=0.085, right=0.70, top=0.9, bottom=0.12)
    style_axes(ax, grid="both")

    ylo, yhi = min(ys) - 0.32, max(ys) + 0.34
    ax.set_ylim(ylo, yhi)
    xpad = (max(xs) - min(xs)) * 0.14 + 0.05
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)

    # role-conflict zone + ceiling
    ax.axhspan(ylo, CEILING, color=FAIL_RED, alpha=0.07, zorder=0)
    ax.axhline(CEILING, color=FAIL_RED, lw=1.5, ls=(0, (5, 3)), zorder=1.5, alpha=0.9)
    ax.text(ax.get_xlim()[0], CEILING - 0.04, "  98.36% role-conflict ceiling",
            ha="left", va="top", fontsize=8.4, color=FAIL_RED, style="italic")

    for i in range(n):
        if i == pidx:
            ax.scatter(xs[i], ys[i], s=460, c=PROPOSED_COLOR, marker="*",
                       edgecolors=PROPOSED_EDGE, linewidth=1.2, zorder=6)
        else:
            ax.scatter(xs[i], ys[i], s=150, c=BASELINE_COLOR,
                       edgecolors=BASELINE_EDGE, linewidth=0.9, zorder=4)

    # evenly-spaced label column on the right, linked by leader lines
    order = sorted(range(n), key=lambda i: ys[i])
    slots = np.linspace(0.05, 0.95, n)
    for slot, i in zip(slots, order):
        is_p = i == pidx
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xycoords="data",
                    xytext=(1.045, slot), textcoords="axes fraction",
                    ha="left", va="center",
                    fontsize=9.6 if is_p else 8.7,
                    color=PROPOSED_EDGE if is_p else SUBTLE_INK,
                    fontweight="bold" if is_p else "normal",
                    arrowprops=dict(arrowstyle="-", color="#C4C4C4", lw=0.7,
                                    shrinkA=4, shrinkB=3,
                                    connectionstyle="arc3,rad=0.0"))

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Subtype accuracy (%)")
    ax.set_title("Ablation cells — parameter count vs. subtype accuracy")

    star = ax.scatter([], [], marker="*", s=300, c=PROPOSED_COLOR,
                      edgecolors=PROPOSED_EDGE, linewidth=1.0,
                      label="proposed (Triplet → SE)")
    cell = ax.scatter([], [], s=130, c=BASELINE_COLOR, edgecolors=BASELINE_EDGE,
                      linewidth=0.9, label="ablation cell")
    legend_clean(ax, handles=[star, cell], loc="lower right")
    save_fig(fig, "05_ablation", "fig05c_ablation_params_vs_acc", tight=False)


if __name__ == "__main__":
    main()
