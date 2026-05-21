"""fig05d — v1 → v2 progression: 4 milestone variants."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        PROPOSED_COLOR, PROPOSED_EDGE, FAIL_RED, MUTED, INK)
from _lib.data_loader import collect_ablation, collect_v2

CEILING = 98.36
SUB_C = "#5E8AA8"
BIN_C = "#C8CDD6"


def main():
    apply_rc()
    rows = {r["variant"]: r for r in collect_ablation()}
    v2 = collect_v2()

    milestones = [
        ("none\n(no AttentionHub)", rows.get("none"), False),
        ("full v1\n(BAM+Triplet+KAN)", rows.get("full"), False),
        ("triplet+KAN\n(best v1 cell)", rows.get("triplet_kan"), False),
        ("Triplet → SE\n(v2, proposed)", v2, True),
    ]
    labels = [m[0] for m in milestones]
    is_prop = [m[2] for m in milestones]
    bins = [m[1]["binary_acc"] * 100 if m[1] else 0 for m in milestones]
    subs = [m[1]["subtype_acc"] * 100 if m[1] else 0 for m in milestones]

    fig, ax = plt.subplots(figsize=(11, 6.0))
    x = np.arange(len(labels))
    w = 0.36

    ax.axhspan(98.0, CEILING, color=FAIL_RED, alpha=0.06, zorder=0)
    ax.axhline(CEILING, color=FAIL_RED, lw=1.4, ls=(0, (5, 3)), zorder=1.5, alpha=0.9)

    b1 = ax.bar(x - w / 2, bins, w, label="Binary accuracy", color=BIN_C,
                edgecolor="none", zorder=3)
    sub_fc = [PROPOSED_COLOR if p else SUB_C for p in is_prop]
    sub_ec = [PROPOSED_EDGE if p else "none" for p in is_prop]
    b2 = ax.bar(x + w / 2, subs, w, label="Subtype accuracy", color=sub_fc,
                edgecolor=sub_ec, lw=0.8, zorder=3)

    for b, v in zip(b1, bins):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8.2, color=MUTED)
    for b, v, p in zip(b2, subs, is_prop):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8.6,
                    color=PROPOSED_EDGE if p else INK,
                    fontweight="bold" if p else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(98.0, 99.95)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("AttentionHub design journey — none → full v1 → best v1 cell → v2 winner")
    ax.text(ax.get_xlim()[1], CEILING - 0.035, "98.36% ceiling  ", ha="right",
            va="top", fontsize=8.5, color=FAIL_RED, style="italic")
    style_axes(ax, grid="y")
    for lbl, p in zip(ax.get_xticklabels(), is_prop):
        if p:
            lbl.set_color(PROPOSED_EDGE)
            lbl.set_fontweight("bold")
    legend_clean(ax, loc="upper left")
    save_fig(fig, "05_ablation", "fig05d_v1_to_v2_progression")


if __name__ == "__main__":
    main()
