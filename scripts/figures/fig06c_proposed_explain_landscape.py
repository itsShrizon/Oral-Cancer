"""fig06c-landscape — 2-column × 3-row variant of the curated showcase.

Each cell contains one curated sample with its triplet stacked vertically:
[label/note above, then 3 columns Original|GradCAM|LIME side-by-side].
Produces a more landscape-shaped figure that suits two-column journal layouts.
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


SAMPLES = [
    ("binary",  3,  4, "Malignant lesion",
                       "Tight red focal spot on the ulcerated region"),
    ("subtype", 1, 14, "CoS — lip lesion",
                       "Sharp focal heat-map on the herpes-like vesicle"),
    ("subtype", 2, 14, "CoS — perioral skin",
                       "Elongated heat-map traces the linear lesion contour"),
    ("subtype", 9, 14, "OC — raised tongue lesion",
                       "Strong activation on the papillomatous lesion"),
    ("subtype", 3, 14, "Gum — inflamed lip margin",
                       "Heat-map concentrates on the inflamed lower lip"),
    ("subtype",10, 14, "OLP — striations / patches",
                       "Two focal spots align with the visible patches"),
]

COL_TITLES = ["Original", "GradCAM++", "LIME"]


def main():
    apply_rc()
    base = REPO_ROOT / "results" / PROPOSED_V2
    src_imgs = {
        "binary": Image.open(base / "explain_binary.png"),
        "subtype": Image.open(base / "explain_subtype.png"),
    }

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "Explainability Showcase  —  Custom EfficientNet V2  (Hub v2, proposed)",
        fontsize=20, fontweight="bold", color="#1B2631", y=0.992,
    )
    fig.text(
        0.5, 0.962,
        "Hand-curated samples where GradCAM++ peaks coincide with the visible "
        "lesion and LIME contours trace its boundary.",
        ha="center", fontsize=12.5, color="#555555", style="italic",
    )

    # Outer grid: 3 rows × 2 columns (each cell is a sample)
    outer = fig.add_gridspec(3, 2,
                             left=0.025, right=0.99,
                             top=0.935, bottom=0.015,
                             hspace=0.18, wspace=0.06)

    for i, (kind, src_row, n_src_rows, cls_label, focus_note) in enumerate(SAMPLES):
        r, c = divmod(i, 2)
        # Inner: 2 rows (header strip, image strip) × 3 cols (orig/cam/lime)
        inner = outer[r, c].subgridspec(2, N_COLS,
                                        height_ratios=[0.18, 1.0],
                                        hspace=0.03, wspace=0.04)
        # Header strip spanning all 3 columns
        hax = fig.add_subplot(inner[0, :])
        hax.axis("off")
        hax.text(0.02, 0.7, cls_label, ha="left", va="center",
                 fontsize=13, fontweight="bold", color="#1B2631",
                 transform=hax.transAxes)
        hax.text(0.02, 0.2,
                 "Focus:  " + focus_note,
                 ha="left", va="center",
                 fontsize=10.5, color=PROPOSED_EDGE, style="italic",
                 transform=hax.transAxes)
        # 3 image columns
        img = src_imgs[kind]
        for c2 in range(N_COLS):
            ax = fig.add_subplot(inner[1, c2])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            tile = crop_image_tile(img, row=src_row, col=c2, n_rows=n_src_rows)
            ax.imshow(tile)
            # Mini column label under each tile (only on top row to save space)
            if r == 0:
                ax.set_title(COL_TITLES[c2], fontsize=10.5, color="#444444",
                             pad=3)

    save_fig(fig, "06_explainability",
             "fig06c_proposed_explain_panel_landscape", tight=False)


if __name__ == "__main__":
    main()
