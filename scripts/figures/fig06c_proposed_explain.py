"""fig06c — GradCAM++ explainability HERO showcase for the proposed model.

Method decision (see fig06d): across all 18 explainability tiles of the
proposed model, GradCAM++ produced tight, lesion-aligned focal hot-spots,
whereas LIME super-pixel boundaries were consistently noisy and over-broad.
GradCAM++ is therefore the selected explainability method for the showcase.

This figure presents the 8 best GradCAM++ samples (ranked by lesion-focus
accuracy) as Original -> GradCAM++ pairs in a 4 x 2 grid.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from _lib.style import apply_rc, save_fig, PROPOSED_COLOR, PROPOSED_EDGE
from _lib.data_loader import REPO_ROOT, PROPOSED_V2


N_COLS = 3
TOP_TITLE_FRAC = 0.014
ROW_LABEL_FRAC = 0.13
COL_ORIGINAL = 0
COL_GRADCAM = 1


def crop_image_tile(img, row, col, n_rows, n_cols=N_COLS):
    w, h = img.size
    title_h = int(h * TOP_TITLE_FRAC)
    body_h = h - title_h
    row_h = body_h / n_rows
    col_w = w / n_cols
    label_h = int(row_h * ROW_LABEL_FRAC)
    y0 = title_h + int(row * row_h) + label_h
    y1_max = title_h + int((row + 1) * row_h)
    y1 = min(y0 + int(col_w * 0.75), y1_max)
    x0 = int(col * col_w)
    x1 = int((col + 1) * col_w)
    return img.crop((x0, y0, x1, y1))


# 8 best GradCAM++ samples, ranked by lesion-focus accuracy (visual scoring).
SAMPLES = [
    ("subtype", 1, 14, "CoS  -  lip vesicle",
                       "Single sharp focal dot exactly on the lesion"),
    ("subtype",10, 14, "OLP  -  bilateral patches",
                       "Two focal spots, one per visible patch"),
    ("subtype", 9, 14, "OC  -  raised tongue lesion",
                       "Strong activation over the papillomatous lesion"),
    ("binary",  3,  4, "Malignant  -  ulcerated region",
                       "Focal hot-spot on the malignant ulcer"),
    ("subtype", 3, 14, "Gum  -  inflamed lip margin",
                       "Heat concentrates on the inflamed lower lip"),
    ("subtype", 2, 14, "CoS  -  perioral skin",
                       "Heat-map traces the linear lesion"),
    ("binary",  1,  4, "Benign  -  tongue tip",
                       "Compact focal response at the tongue tip"),
    ("subtype",12, 14, "OT  -  tongue surface",
                       "Focal activation on the affected tongue tip"),
]


def main():
    apply_rc()
    base = REPO_ROOT / "results" / PROPOSED_V2
    src_imgs = {
        "binary":  Image.open(base / "explain_binary.png"),
        "subtype": Image.open(base / "explain_subtype.png"),
    }

    n_pairs = len(SAMPLES)        # 8
    grid_rows, grid_cols = 4, 2   # 4 x 2 layout of sample pairs

    fig = plt.figure(figsize=(16, 19))

    fig.suptitle(
        "GradCAM++ Explainability Showcase  —  Custom EfficientNet V2 (Hub v2)",
        fontsize=19, fontweight="bold", color="#1B2631", y=0.988,
    )
    fig.text(
        0.5, 0.963,
        "GradCAM++ selected over LIME: it yields tight, lesion-aligned focal "
        "hot-spots (see method-comparison figure).  8 best samples, ranked by "
        "lesion-focus accuracy.",
        ha="center", fontsize=12, color="#555555", style="italic",
    )

    # Outer grid: 4 rows x 2 cols of sample cells
    outer = fig.add_gridspec(grid_rows, grid_cols,
                             left=0.035, right=0.985,
                             top=0.945, bottom=0.015,
                             hspace=0.30, wspace=0.12)

    for i, (kind, src_row, n_src, cls_label, note) in enumerate(SAMPLES):
        r, c = divmod(i, grid_cols)
        # Inner: header strip + 2 image columns (Original | GradCAM++)
        inner = outer[r, c].subgridspec(2, 2, height_ratios=[0.20, 1.0],
                                        hspace=0.04, wspace=0.04)
        # Header
        hax = fig.add_subplot(inner[0, :])
        hax.axis("off")
        hax.text(0.0, 0.68, f"{i+1}.  {cls_label}", ha="left", va="center",
                 fontsize=13, fontweight="bold", color="#1B2631",
                 transform=hax.transAxes)
        hax.text(0.0, 0.18, "Focus:  " + note, ha="left", va="center",
                 fontsize=10.5, color=PROPOSED_EDGE, style="italic",
                 transform=hax.transAxes)

        img = src_imgs[kind]
        for j, (col_idx, col_name) in enumerate([(COL_ORIGINAL, "Original"),
                                                 (COL_GRADCAM, "GradCAM++")]):
            ax = fig.add_subplot(inner[1, j])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            tile = crop_image_tile(img, row=src_row, col=col_idx, n_rows=n_src)
            ax.imshow(tile)
            ax.set_xlabel(col_name, fontsize=10.5, color="#444444", labelpad=3)
            # Highlight the GradCAM++ tile with a colored frame
            if col_idx == COL_GRADCAM:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(PROPOSED_COLOR)
                    s.set_linewidth(2.2)

    save_fig(fig, "06_explainability", "fig06c_proposed_explain_panel", tight=False)


if __name__ == "__main__":
    main()
