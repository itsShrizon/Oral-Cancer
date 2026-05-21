"""fig02 — Dataset sample grid (DS1 binary + DS2 7-class subtype)."""
from __future__ import annotations

import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from PIL import Image

from _lib.style import apply_rc, save_fig, INK, MUTED
from _lib.data_loader import REPO_ROOT


N_PER_CLASS = 4
GREEN, RED, SLATE = "#1F8F6E", "#B5413B", "#5B7CA8"


def pick(folder: Path, n: int, rng: random.Random):
    if not folder.exists():
        return []
    cand = sorted([f for f in folder.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    if not cand:
        return []
    return rng.sample(cand, min(n, len(cand)))


def main():
    apply_rc()
    rng = random.Random(42)

    ds1_root = REPO_ROOT / "Dataset 1" / "original_data"
    ds2_root = REPO_ROOT / "Dataset 2" / "Training"

    rows = []
    rows.append(("Benign", "DS1", pick(ds1_root / "benign_lesions", N_PER_CLASS, rng), GREEN))
    rows.append(("Malignant", "DS1", pick(ds1_root / "malignant_lesions", N_PER_CLASS, rng), RED))
    for cls in ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]:
        rows.append((cls, "DS2", pick(ds2_root / cls, N_PER_CLASS, rng), SLATE))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, N_PER_CLASS,
                             figsize=(N_PER_CLASS * 2.45, nrows * 2.05))

    for r, (name, ds, files, chip) in enumerate(rows):
        for c in range(N_PER_CLASS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color("#D9D9D9")
                s.set_linewidth(0.8)
            if c < len(files):
                try:
                    img = Image.open(files[c]).convert("RGB").resize((420, 420))
                    ax.imshow(img)
                except Exception as e:  # noqa: BLE001
                    ax.set_facecolor("#F2F2F2")
                    ax.text(0.5, 0.5, "[error]", ha="center", va="center",
                            fontsize=8, color="#999999", transform=ax.transAxes)
            else:
                ax.set_facecolor("#F4F4F4")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=10, color="#AAAAAA", transform=ax.transAxes)

    fig.subplots_adjust(left=0.16, right=0.996, top=0.963, bottom=0.006,
                        wspace=0.035, hspace=0.055)

    # ---- row-label chips in the left gutter ----
    chip_x = 0.088
    for r, (name, ds, files, chip) in enumerate(rows):
        pos = axes[r, 0].get_position()
        yc = (pos.y0 + pos.y1) / 2
        fig.text(chip_x, yc, name, ha="center", va="center", fontsize=11,
                 fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor=chip,
                           edgecolor="none"))

    # ---- section group labels + divider ----
    p0 = axes[0, 0].get_position()
    p1 = axes[1, 0].get_position()
    p2 = axes[2, 0].get_position()
    pN = axes[nrows - 1, 0].get_position()
    sec_x = 0.032
    fig.text(sec_x, (p0.y1 + p1.y0) / 2, "D A T A S E T  1", rotation=90,
             ha="center", va="center", fontsize=9, fontweight="bold",
             color=MUTED)
    fig.text(sec_x, (p2.y1 + pN.y0) / 2, "D A T A S E T  2", rotation=90,
             ha="center", va="center", fontsize=9, fontweight="bold",
             color=MUTED)
    y_sep = (p1.y0 + p2.y1) / 2
    fig.add_artist(mlines.Line2D([0.012, 0.15], [y_sep, y_sep],
                                 color="#C4C4C4", lw=1.0,
                                 linestyle=(0, (4, 3)),
                                 transform=fig.transFigure))

    fig.suptitle("Dataset overview — four sample images per class (seed = 42)",
                 x=0.012, ha="left", fontsize=13, fontweight="bold",
                 color=INK, y=0.992)
    save_fig(fig, "02_dataset", "fig02_dataset_sample_grid", tight=False)


if __name__ == "__main__":
    main()
