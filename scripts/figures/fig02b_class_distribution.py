"""fig02b — Class distribution: DS1 binary counts + DS2 test-set support."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from _lib.style import (apply_rc, save_fig, style_axes, legend_clean,
                        INK, MUTED, SUBTLE_INK)
from _lib.data_loader import REPO_ROOT, parse_per_class, PROPOSED_V2

BENIGN_C = "#3C8DBC"
MALIGN_C = "#D55E00"
TOTAL_C = "#C2C7D0"
SUPPORT_C = "#3C8DBC"


def count_dir(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir()
               if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


def count_ds2_total(cls: str) -> int:
    root = REPO_ROOT / "Dataset 2"
    total = 0
    for split in ["Training", "Validation", "Testing"]:
        total += count_dir(root / split / cls)
    return total


def main():
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9),
                             gridspec_kw={"width_ratios": [1.0, 2.0]})

    # ---- DS1 binary ----
    ds1 = REPO_ROOT / "Dataset 1" / "original_data"
    n_ben = count_dir(ds1 / "benign_lesions")
    n_mal = count_dir(ds1 / "malignant_lesions")
    ax = axes[0]
    vals = [n_ben, n_mal]
    bars = ax.bar(["Benign", "Malignant"], vals, width=0.6,
                  color=[BENIGN_C, MALIGN_C], edgecolor="none")
    ax.set_ylim(0, max(vals) * 1.16)
    for b, v in zip(bars, vals):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3.5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold", color=SUBTLE_INK)
    ax.set_title("Dataset 1 — binary supervision")
    ax.set_ylabel("Image count")
    style_axes(ax, grid="y")

    # ---- DS2 subtype: total images + test support ----
    classes = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
    totals = [count_ds2_total(c) for c in classes]
    per_class = parse_per_class(PROPOSED_V2) or {}
    supports = [per_class.get(c, {}).get("support", 0) for c in classes]

    ax = axes[1]
    x = np.arange(len(classes))
    w = 0.38
    b1 = ax.bar(x - w / 2, totals, w, label="Total (Train + Val + Test)",
                color=TOTAL_C, edgecolor="none")
    b2 = ax.bar(x + w / 2, supports, w, label="Test-set support",
                color=SUPPORT_C, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_ylabel("Image count")
    ax.set_title("Dataset 2 — subtype supervision  (7 classes)")
    for bars_, vals_ in [(b1, totals), (b2, supports)]:
        for b, v in zip(bars_, vals_):
            if v:
                ax.annotate(str(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8.3, color=MUTED)
    style_axes(ax, grid="y")
    legend_clean(ax, loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2)

    fig.suptitle("Class distribution across the two datasets", x=0.012, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "02_dataset", "fig02b_class_distribution", tight=False)


if __name__ == "__main__":
    main()
