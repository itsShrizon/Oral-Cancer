"""fig04b — Per-class precision/recall/F1 grouped bars for the proposed model."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import apply_rc, save_fig, style_axes, legend_clean, MUTED
from _lib.data_loader import parse_per_class, PROPOSED_V2


CLASSES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
C_PREC, C_REC, C_F1 = "#0072B2", "#009E73", "#D55E00"


def main():
    apply_rc()
    pc = parse_per_class(PROPOSED_V2) or {}
    prec = [pc.get(c, {}).get("precision", 0) * 100 for c in CLASSES]
    rec = [pc.get(c, {}).get("recall", 0) * 100 for c in CLASSES]
    f1 = [pc.get(c, {}).get("f1", 0) * 100 for c in CLASSES]
    sup = [pc.get(c, {}).get("support", 0) for c in CLASSES]

    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    x = np.arange(len(CLASSES))
    w = 0.26
    series = [(prec, "Precision", C_PREC, -w),
              (rec, "Recall", C_REC, 0.0),
              (f1, "F1", C_F1, w)]
    for vals, label, color, off in series:
        bars = ax.bar(x + off, vals, w, label=label, color=color, edgecolor="none")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2.4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.2, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn = {s}" for c, s in zip(CLASSES, sup)])
    ax.set_ylim(96.0, 101.0)
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-class precision / recall / F1 — proposed Custom V2 (Hub v2)")
    style_axes(ax, grid="y")
    legend_clean(ax, loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3)
    save_fig(fig, "04_per_class", "fig04b_per_class_metrics_proposed")


if __name__ == "__main__":
    main()
