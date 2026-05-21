"""fig04 — Per-class F1 heatmap (models x 7 subtype classes).

Source: parsed sklearn classification_report blocks from
results/<run>/evaluation_results.txt (Table 2a in paper).
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import (apply_rc, save_fig, MODEL_LABELS, PROPOSED_COLOR,
                        PROPOSED_EDGE, INK, MUTED, HAIRLINE)
from _lib.data_loader import (parse_per_class, BASELINE_RUNS, PROPOSED_V1, PROPOSED_V2)


CLASSES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]


def main():
    apply_rc()
    runs = [
        "resnet50", "densenet121", "efficientnet_b0",
        "efficientnet_v2b2", "efficientnet_v2b3",
        "inception_v3", PROPOSED_V1, PROPOSED_V2,
    ]
    matrix, labels, runkeys = [], [], []
    for run in runs:
        pc = parse_per_class(run)
        if pc is None:
            continue
        matrix.append([pc.get(c, {}).get("f1", np.nan) for c in CLASSES])
        labels.append(MODEL_LABELS.get(run, run))
        runkeys.append(run)
    M = np.array(matrix) * 100  # percent

    fig, ax = plt.subplots(figsize=(9.6, 5.9))
    vmin = max(90.0, float(np.nanmin(M)) - 0.4)
    vmax = 100.0
    im = ax.imshow(M, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")

    # crisp white separators between cells
    ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Per-class F1 (%)", fontsize=10, color=MUTED)
    cb.outline.set_edgecolor(HAIRLINE)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=MUTED, length=3)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            norm = (v - vmin) / (vmax - vmin)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if norm > 0.62 else INK)

    # highlight the proposed row
    for i, run in enumerate(runkeys):
        if run == PROPOSED_V2:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(CLASSES), 1,
                                       fill=False, edgecolor=PROPOSED_COLOR,
                                       lw=2.4, zorder=5))
            ax.get_yticklabels()[i].set_color(PROPOSED_EDGE)
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_title("Per-class F1 score across models  (Table 2a)")
    ax.set_xlabel("Subtype class")
    save_fig(fig, "04_per_class", "fig04_per_class_f1_heatmap")


if __name__ == "__main__":
    main()
