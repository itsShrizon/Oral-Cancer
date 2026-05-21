"""fig05 — Ablation bar chart with 98.36% role-conflict ceiling line."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        PROPOSED_COLOR, PROPOSED_EDGE, FAIL_RED, INK, MUTED,
                        ABLATION_LABELS)
from _lib.data_loader import collect_ablation, collect_v2

CEILING = 98.36
SUB_C = "#5E8AA8"   # neutral slate for subtype bars
BIN_C = "#C8CDD6"   # light gray companion bars
XMIN = 97.8


def main():
    apply_rc()
    data = []
    for r in collect_ablation():
        data.append({"label": ABLATION_LABELS.get(r["variant"], r["variant"]),
                     "binary": r["binary_acc"] * 100,
                     "subtype": r["subtype_acc"] * 100, "proposed": False})
    v2 = collect_v2()
    if v2:
        data.append({"label": "Triplet → SE  (v2, proposed)",
                     "binary": v2["binary_acc"] * 100,
                     "subtype": v2["subtype_acc"] * 100, "proposed": True})
    data.sort(key=lambda d: d["subtype"], reverse=True)

    labels = [d["label"] for d in data]
    bins = [d["binary"] for d in data]
    subs = [d["subtype"] for d in data]
    n = len(data)

    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    y = np.arange(n)
    h = 0.38

    # role-conflict zone + ceiling line
    ax.axvspan(XMIN, CEILING, color=FAIL_RED, alpha=0.07, zorder=0)
    ax.axvline(CEILING, color=FAIL_RED, lw=1.5, ls=(0, (5, 3)), zorder=2, alpha=0.9)

    sub_fc = [PROPOSED_COLOR if d["proposed"] else SUB_C for d in data]
    sub_ec = [PROPOSED_EDGE if d["proposed"] else "none" for d in data]
    bars_sub = ax.barh(y - h / 2, subs, h, color=sub_fc, edgecolor=sub_ec,
                       lw=0.8, zorder=3, label="Subtype accuracy")
    bars_bin = ax.barh(y + h / 2, bins, h, color=BIN_C, edgecolor="none",
                       zorder=3, label="Binary accuracy")

    for d, b, v in zip(data, bars_sub, subs):
        ax.annotate(f"{v:.2f}", (v, b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=8.6,
                    color=PROPOSED_EDGE if d["proposed"] else INK,
                    fontweight="bold" if d["proposed"] else "normal")
    for b, v in zip(bars_bin, bins):
        ax.annotate(f"{v:.2f}", (v, b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=8.0, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(XMIN, 100.0)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("AttentionHub ablation — subtype vs. binary accuracy across variants  (Table 4 + v2)")
    ax.text(CEILING, -0.78, "  98.36% role-conflict ceiling", ha="left",
            va="center", fontsize=8.6, color=FAIL_RED, style="italic")
    style_axes(ax, grid="x")
    for lbl, d in zip(ax.get_yticklabels(), data):
        if d["proposed"]:
            lbl.set_color(PROPOSED_EDGE)
            lbl.set_fontweight("bold")
    legend_clean(ax, loc="lower right")
    save_fig(fig, "05_ablation", "fig05_ablation_bars_with_ceiling")


if __name__ == "__main__":
    main()
