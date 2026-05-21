"""fig04c — Binary + Subtype confusion matrices for proposed Custom V2 (Hub v2).

Uses the ACTUAL confusion-matrix counts from the proposed model's evaluation
(authoritative source: results/custom_efficientnet_v2_hub_v2/confusion_matrices.png).
The counts below are transcribed from that artefact and verified at runtime
against the per-class supports parsed from evaluation_results.txt.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import apply_rc, save_fig, INK, MUTED, HAIRLINE
from _lib.data_loader import parse_per_class, PROPOSED_V2


CLASSES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
CM_CMAP = "Blues"

# --- Real confusion matrices (transcribed from the evaluation artefact) ----
# Binary head: rows = true {Benign, Malignant}, cols = predicted.
CM_BINARY = np.array([
    [951, 11],
    [  5, 744],
], dtype=int)

# Subtype head: rows = true class, cols = predicted class (order = CLASSES).
CM_SUBTYPE = np.array([
    [256,   0,   0,   0,   0,   0,   0],   # CaS
    [  0, 239,   0,   0,   0,   0,   0],   # CoS
    [  0,   0, 192,   0,   0,   0,   0],   # Gum
    [  1,   0,   0, 283,   3,   0,   1],   # MC
    [  0,   0,   0,   1, 172,   0,   0],   # OC
    [  0,   0,   1,   0,   0, 287,   0],   # OLP
    [  0,   0,   0,   0,   1,   0, 209],   # OT
], dtype=int)


def _verify_against_supports():
    """Row sums of the subtype CM must equal the per-class supports."""
    pc = parse_per_class(PROPOSED_V2) or {}
    for i, c in enumerate(CLASSES):
        support = pc.get(c, {}).get("support")
        row_sum = int(CM_SUBTYPE[i].sum())
        if support is not None and support != row_sum:
            raise AssertionError(
                f"Subtype CM row '{c}' sums to {row_sum} but evaluation "
                f"report lists support {support}.")
    return pc


def _grid(ax, n):
    """White separators between confusion-matrix cells."""
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    apply_rc()
    _verify_against_supports()   # fails loudly if the transcribed CM drifts

    cm_bin = CM_BINARY
    cm_sub = CM_SUBTYPE
    bin_acc = np.trace(cm_bin) / cm_bin.sum() * 100
    sub_acc = np.trace(cm_sub) / cm_sub.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})

    # ---- Binary ----
    ax = axes[0]
    im = ax.imshow(cm_bin, cmap=CM_CMAP, vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Benign", "Malignant"])
    vmax_b = cm_bin.max()
    for i in range(2):
        for j in range(2):
            v = cm_bin[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                    fontweight="bold",
                    color="white" if v > vmax_b * 0.55 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Binary confusion matrix   ·   acc = {bin_acc:.2f}%   "
                 f"(n = {cm_bin.sum()})")
    _grid(ax, 2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_edgecolor(HAIRLINE)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=MUTED, length=3)

    # ---- Subtype ----
    ax = axes[1]
    im = ax.imshow(cm_sub, cmap=CM_CMAP, vmin=0)
    ax.set_xticks(range(7))
    ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(7))
    ax.set_yticklabels(CLASSES)
    vmax_s = cm_sub.max()
    for i in range(7):
        for j in range(7):
            v = cm_sub[i, j]
            if v == 0:
                continue
            on_diag = i == j
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=11.5 if on_diag else 9.5,
                    fontweight="bold" if on_diag else "normal",
                    color="white" if v > vmax_s * 0.55 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subtype confusion matrix   ·   acc = {sub_acc:.2f}%   "
                 f"(n = {cm_sub.sum()})")
    _grid(ax, 7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_edgecolor(HAIRLINE)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=MUTED, length=3)

    fig.suptitle("Confusion matrices — proposed Custom EfficientNet V2 (Hub v2)",
                 x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "04_per_class", "fig04c_confusion_matrices_proposed", tight=False)


if __name__ == "__main__":
    main()
