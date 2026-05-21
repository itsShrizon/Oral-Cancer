"""fig06 — Cross-model GradCAM++ comparison on the SAME 4 binary samples.

Source files all share deterministic seed/split, so row i in
results/<model>/explain_binary.png is the same input image across models.
Layout per source: 3 columns (Original | GradCAM++ | LIME) and N rows.
We crop just the image portion of the GradCAM++ tile (col 1), skipping the
per-tile label strip above each image.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from _lib.style import apply_rc, save_fig, MODEL_LABELS, PROPOSED_COLOR
from _lib.data_loader import REPO_ROOT, PROPOSED_V2


MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_v2b2",
    "inception_v3",
    PROPOSED_V2,
]
N_SAMPLES = 4   # 4 binary samples per source file
N_COLS = 3      # source layout: Original | GradCAM++ | LIME
COL_GRADCAM = 1
COL_LIME = 2
TOP_TITLE_FRAC = 0.014    # top ~1.4% is the global title strip
ROW_LABEL_FRAC = 0.13     # skip per-tile label strip at top of each row


def crop_image_tile(img: Image.Image, row: int, col: int, n_rows: int,
                    n_cols: int = N_COLS):
    """Return the image region inside one tile, excluding label strips."""
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


def build_grid(col_idx: int, title: str, out_name: str, source: str = "binary"):
    """source: 'binary' (4 samples, square-ish) or 'subtype' (14 samples)."""
    apply_rc()
    n_rows_grid = len(MODELS)
    n_cols_grid = N_SAMPLES

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                             figsize=(n_cols_grid * 2.8, n_rows_grid * 2.9))
    if n_rows_grid == 1:
        axes = np.array([axes])

    # Sample labels (binary: 2 Benign + 2 Malignant per explain_model.py)
    sample_labels = ["Benign #1", "Benign #2", "Malignant #1", "Malignant #2"]

    for r, run in enumerate(MODELS):
        src = REPO_ROOT / "results" / run / f"explain_{source}.png"
        if not src.exists():
            for c in range(n_cols_grid):
                axes[r, c].axis("off")
            continue
        img = Image.open(src)
        n_rows_src = N_SAMPLES if source == "binary" else 14
        for c in range(n_cols_grid):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= n_rows_src:
                ax.set_facecolor("#F2F2F2")
                continue
            tile = crop_image_tile(img, row=c, col=col_idx, n_rows=n_rows_src)
            ax.imshow(tile)
            if r == 0:
                ax.set_title(sample_labels[c], fontsize=10.5, fontweight="bold",
                             color="#1B2631", pad=6)
            if c == 0:
                lbl = MODEL_LABELS.get(run, run)
                is_prop = run == PROPOSED_V2
                ax.text(-0.10, 0.5,
                        lbl + ("\n(proposed)" if is_prop else ""),
                        ha="right", va="center",
                        fontsize=11.5 if is_prop else 11,
                        fontweight="bold" if is_prop else "regular",
                        color=PROPOSED_COLOR if is_prop else "#1B2631",
                        transform=ax.transAxes)
            # Highlight proposed model row with a colored border
            if run == PROPOSED_V2:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(PROPOSED_COLOR)
                    s.set_linewidth(2.5)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.subplots_adjust(wspace=0.05, hspace=0.18, left=0.10, right=0.99,
                        top=0.93, bottom=0.02)
    save_fig(fig, "06_explainability", out_name, tight=False)


def main():
    build_grid(col_idx=COL_GRADCAM,
               title="GradCAM++ on the same 4 binary samples - cross-model comparison",
               out_name="fig06_gradcam_cross_model_composite")


if __name__ == "__main__":
    main()
