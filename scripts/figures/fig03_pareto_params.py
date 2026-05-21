"""fig03 / fig03b — accuracy-vs-cost Pareto plots.

`pareto_figure` is the shared renderer; fig03b imports it. The y-axis is zoomed
to the main accuracy band; any model that falls far below it is shown as an
off-axis marker so it is not silently dropped.
"""
from __future__ import annotations

import math
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt

from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        PROPOSED_COLOR, PROPOSED_EDGE, BASELINE_COLOR,
                        BASELINE_EDGE, SUBTLE_INK, MUTED)
from _lib.data_loader import collect_table1, PROPOSED_V2
from _lib.pareto import pareto_front
from _lib.labels import smart_annotate


def pareto_figure(xkey, xlabel, ykey, ylabel, title, out_name, log_x=True):
    apply_rc()
    rows = collect_table1()
    xs = [r[xkey] for r in rows]
    ys = [r[ykey] * 100.0 for r in rows]
    n = len(rows)
    pidx = next(i for i, r in enumerate(rows) if r["run"] == PROPOSED_V2)

    # Split off low outliers so one weak model does not compress the axis.
    med = sorted(ys)[n // 2]
    outliers = [i for i in range(n) if ys[i] < med - 4.0]
    inrange = [i for i in range(n) if i not in outliers]

    front = sorted(pareto_front(xs, ys, minimize_x=True, maximize_y=True),
                   key=lambda i: xs[i])

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    style_axes(ax, grid="both")
    if log_x:
        ax.set_xscale("log")

    ymain = [ys[i] for i in inrange]
    ypad = (max(ymain) - min(ymain)) * 0.16 + 0.35
    ylo, yhi = min(ymain) - ypad, min(100.55, max(ymain) + ypad)
    ax.set_ylim(ylo, yhi)

    # Pareto frontier (behind the markers)
    (frontier,) = ax.plot([xs[i] for i in front], [ys[i] for i in front],
                          color="#A9B1BF", lw=1.7, ls=(0, (6, 3)), zorder=1.6,
                          label="Pareto frontier")

    # baseline markers — uniform muted slate
    for i in inrange:
        if i == pidx:
            continue
        ax.scatter(xs[i], ys[i], s=145, c=BASELINE_COLOR, edgecolors=BASELINE_EDGE,
                   linewidth=0.9, zorder=4)
    # proposed model — accent star
    star = ax.scatter(xs[pidx], ys[pidx], s=470, c=PROPOSED_COLOR,
                      edgecolors=PROPOSED_EDGE, linewidth=1.2, marker="*",
                      zorder=6, label="Proposed — Custom V2 (Hub v2)")

    # off-axis outliers (e.g. ConvNeXt-T) shown as a downward marker at the floor
    y_floor = ylo + (yhi - ylo) * 0.05
    for i in outliers:
        ax.scatter(xs[i], y_floor, marker="v", s=120, c=BASELINE_COLOR,
                   edgecolors=BASELINE_EDGE, linewidth=0.9, zorder=4)
        ax.annotate(f"{rows[i]['label']}  ·  {ys[i]:.1f}%  (below axis)",
                    (xs[i], y_floor), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.2, color=MUTED,
                    style="italic")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # in-range point labels, collision-free
    labels = [rows[i]["label"] for i in inrange]
    colors = [PROPOSED_EDGE if i == pidx else SUBTLE_INK for i in inrange]
    weights = ["bold" if i == pidx else "normal" for i in inrange]
    fsizes = [10.0 if i == pidx else 8.7 for i in inrange]

    base_r = math.sqrt(145 / math.pi) * 120 / 72
    star_r = math.sqrt(470 / math.pi) * 120 / 72 * 0.82
    radii = [star_r if i == pidx else base_r for i in inrange]

    fig.tight_layout()
    smart_annotate(ax, [xs[i] for i in inrange], [ys[i] for i in inrange],
                   labels, fontsize=fsizes, colors=colors, weights=weights,
                   prefer="up", point_radii=radii)

    legend_clean(ax, handles=[star, frontier], loc="lower left",
                 fontsize=9.3, handletextpad=0.7, borderaxespad=0.9)
    save_fig(fig, "03_benchmark", out_name, tight=False)


def main():
    pareto_figure(
        xkey="params_m", xlabel="Parameters (M, log scale)",
        ykey="subtype_acc", ylabel="Subtype accuracy (%)",
        title="Subtype accuracy vs. parameter count",
        out_name="fig03_pareto_params", log_x=True,
    )


if __name__ == "__main__":
    main()
