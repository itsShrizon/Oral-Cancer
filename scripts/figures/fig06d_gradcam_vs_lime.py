"""fig06d — Method comparison: why GradCAM++ was chosen over LIME.

Across all 18 explainability tiles of the proposed model, GradCAM++ produced
tight, lesion-aligned focal hot-spots; LIME super-pixel boundaries were
consistently noisy and over-broad. This figure shows the head-to-head on four
representative samples (Original | GradCAM++ | LIME) and tallies the verdict.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from _lib.style import (apply_rc, save_fig, PROPOSED_COLOR, PROPOSED_EDGE,
                        GOOD_GREEN, FAIL_RED, MUTED)
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


# Four representative head-to-head samples.
SAMPLES = [
    ("subtype", 1, 14, "CoS - lip vesicle"),
    ("subtype",10, 14, "OLP - bilateral patches"),
    ("subtype", 3, 14, "Gum - inflamed lip margin"),
    ("binary",  3,  4, "Malignant - ulcerated region"),
]


def main():
    apply_rc()
    base = REPO_ROOT / "results" / PROPOSED_V2
    src = {
        "binary": Image.open(base / "explain_binary.png"),
        "subtype": Image.open(base / "explain_subtype.png"),
    }

    n = len(SAMPLES)
    fig = plt.figure(figsize=(13, 3.4 * n + 2.6))

    fig.suptitle(
        "Method Comparison  —  GradCAM++  vs  LIME  (proposed model)",
        fontsize=18, fontweight="bold", color="#1A1A1A", y=0.985,
    )
    fig.text(
        0.5, 0.957,
        "GradCAM++ gives tight focal hot-spots aligned to the lesion; "
        "LIME super-pixel boundaries are noisy and over-broad.",
        ha="center", fontsize=11.5, color=MUTED, style="italic",
    )

    gs = fig.add_gridspec(
        n + 1, 3,
        height_ratios=[0.26] + [1.0] * n,
        left=0.13, right=0.985, top=0.93, bottom=0.085,
        hspace=0.12, wspace=0.05,
    )

    headers = ["Original", "GradCAM++  (selected)", "LIME  (rejected)"]
    header_colors = ["#1A1A1A", PROPOSED_EDGE, FAIL_RED]
    for c, (htxt, hcol) in enumerate(zip(headers, header_colors)):
        hax = fig.add_subplot(gs[0, c])
        hax.axis("off")
        hax.text(0.5, 0.4, htxt, ha="center", va="center",
                 fontsize=13, fontweight="bold", color=hcol,
                 transform=hax.transAxes)

    for ri, (kind, srow, nsrc, label) in enumerate(SAMPLES, start=1):
        img = src[kind]
        for c in range(3):
            ax = fig.add_subplot(gs[ri, c])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            tile = crop_image_tile(img, row=srow, col=c, n_rows=nsrc)
            ax.imshow(tile)
            if c == 0:
                ax.text(-0.09, 0.5, label, ha="right", va="center",
                        fontsize=11.5, fontweight="bold", color="#1A1A1A",
                        rotation=90, transform=ax.transAxes)
            # Frame GradCAM in green (selected), LIME in red (rejected)
            if c == 1:
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(GOOD_GREEN); s.set_linewidth(2.4)
            elif c == 2:
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(FAIL_RED); s.set_linewidth(2.0)

    # Verdict strip at the bottom
    vax = fig.add_axes([0.13, 0.012, 0.855, 0.055])
    vax.axis("off")
    vax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#F4F8F4", edgecolor=GOOD_GREEN, linewidth=1.4,
        transform=vax.transAxes))
    vax.text(0.5, 0.5,
             "Verdict:  GradCAM++ is the selected explainability method  -  "
             "tight focal localisation on lesions across all 18 inspected "
             "tiles, vs noisy / over-broad super-pixel boundaries from LIME.",
             ha="center", va="center", fontsize=11, color="#1A1A1A",
             fontweight="bold", transform=vax.transAxes)

    save_fig(fig, "06_explainability", "fig06d_gradcam_vs_lime", tight=False)


if __name__ == "__main__":
    main()
