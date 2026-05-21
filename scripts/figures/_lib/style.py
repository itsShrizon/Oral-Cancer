"""Publication-grade matplotlib styling and PNG+PDF export helper.

Minimal-academic aesthetic for an ACL submission: colorblind-safe Okabe-Ito
palette, Segoe UI typography, thin spines, subtle dotted grid, left-aligned
titles, embedded PDF fonts. The proposed model is the only series that carries
an accent color; every baseline is rendered in a muted slate so the hero stands
out at a glance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_ROOT = REPO_ROOT / "figures"

# ---------------------------------------------------------------- core inks
INK = "#1A1A1A"          # near-black for primary text / titles
SUBTLE_INK = "#444444"   # axis labels / tick labels
MUTED = "#6E6E6E"        # secondary / footnote text
HAIRLINE = "#BBBBBB"     # spines
GRID = "#E6E6E6"         # grid lines

# ----------------------------------------------------------- accent palette
# Proposed model — the single accent (Okabe-Ito vermillion).
PROPOSED_COLOR = "#D55E00"
PROPOSED_EDGE = "#8A3D00"
PROPOSED_SOFT = "#FBE3D2"   # pale wash for highlight bands / fills

# Baselines — deliberately desaturated so they recede behind the proposed model.
BASELINE_COLOR = "#8893A8"
BASELINE_EDGE = "#5C6677"

# Status accents (used by ablation pass/fail encodings).
HIGHLIGHT_GOLD = "#E69F00"
GOOD_GREEN = "#179C77"
FAIL_RED = "#B5413B"

# Okabe-Ito colorblind-safe palette (+ a few extras for >8-series plots).
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
    "#D55E00", "#8C6BB1", "#999999", "#117733", "#882255", "#44AA99",
]

# --------------------------------------------- architecture-diagram palette
# Flat, soft fills with matched mid-tone edges.
COLOR_STAGE_FILL = "#E8EEF6"
COLOR_STAGE_EDGE = "#5F7FA8"
COLOR_HUB_FILL = "#FCEBD3"
COLOR_HUB_EDGE = "#C98A3E"
COLOR_ATTN_FILL = "#F6D7B5"
COLOR_ATTN_EDGE = "#AA6B2C"
COLOR_HEAD_FILL = "#E2EEE3"
COLOR_HEAD_EDGE = "#5E9069"
COLOR_IO_FILL = "#EDEDED"
COLOR_IO_EDGE = "#8C8C8C"

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "convnext_tiny": "ConvNeXt-T",
    "swin_t": "Swin-T",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_v2b2": "EffNetV2-B2",
    "efficientnet_v2b3": "EffNetV2-B3",
    "efficientnet_v2s": "EffNetV2-S",
    "inception_v3": "Inception V3",
    "custom_efficientnet_v2_baseline_recipe": "Custom V2 (Hub v1)",
    "custom_efficientnet_v2_hub_v2": "Custom V2 (Hub v2)",
    "custom_efficientnet_v2": "Custom V2 (legacy)",
}

ABLATION_LABELS = {
    "none": "none",
    "bam": "BAM",
    "triplet": "Triplet",
    "kan": "KAN",
    "bam_triplet": "BAM+Triplet",
    "bam_kan": "BAM+KAN",
    "triplet_kan": "Triplet+KAN",
    "full": "BAM+Triplet+KAN (full)",
}


def apply_rc():
    """Apply the minimal-academic rcParams used by every figure."""
    mpl.rcParams.update({
        # canvas
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # embed real fonts in vector output (ACL camera-ready requirement)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "text.color": INK,
        # titles — left-aligned, compact, dark-gray (not pure black)
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10.0,
        # axis labels
        "axes.labelsize": 11,
        "axes.labelcolor": SUBTLE_INK,
        "axes.labelpad": 5.0,
        # spines — hairline, only left + bottom
        "axes.edgecolor": HAIRLINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        # ticks
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        # legend
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.3,
        "legend.labelspacing": 0.45,
        "legend.borderaxespad": 0.4,
        # grid
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": ":",
        "grid.alpha": 1.0,
        # lines / patches
        "lines.linewidth": 1.9,
        "lines.markeredgewidth": 0.8,
        "patch.linewidth": 0.8,
        "hatch.linewidth": 0.6,
    })


def save_fig(fig, category: str, name: str, formats=("png", "pdf"), tight: bool = True):
    """Save figure to figures/<category>/<name>.{png,pdf}."""
    out_dir = FIGURES_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    paths = []
    for ext in formats:
        out = out_dir / f"{name}.{ext}"
        fig.savefig(str(out), bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    rel_paths = [str(p.relative_to(REPO_ROOT)) for p in paths]
    print("Saved " + ", ".join(rel_paths))
    return paths


def style_axes(ax, grid: Optional[str] = "y"):
    """Despine, apply the hairline spines, ticks, and a subtle dotted grid.

    grid: which axis gets gridlines — 'x', 'y', 'both', or None.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        if side in ax.spines:
            ax.spines[side].set_color(HAIRLINE)
            ax.spines[side].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=SUBTLE_INK, length=3.5, width=0.8)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=GRID, linewidth=0.7, linestyle=":", alpha=1.0)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle=":", alpha=1.0)
    if grid not in ("x", "y", "both"):
        ax.grid(False)
    return ax


def add_bar_labels(ax, bars, values=None, fmt="{:.1f}", fontsize=8.6,
                   color=SUBTLE_INK, pad=3.0, weight="normal",
                   horizontal=False):
    """Place compact value labels just past the end of each bar."""
    for i, bar in enumerate(bars):
        if horizontal:
            v = bar.get_width() if values is None else values[i]
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.annotate(fmt.format(v), (x, y), xytext=(pad, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=fontsize, color=color, fontweight=weight,
                        clip_on=False)
        else:
            v = bar.get_height() if values is None else values[i]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.annotate(fmt.format(v), (x, y), xytext=(0, pad),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=fontsize, color=color, fontweight=weight,
                        clip_on=False)


def panel_tag(ax, letter: str, x: float = -0.085, y: float = 1.06,
              fontsize: float = 12.0):
    """Lowercase panel letter, e.g. (a), for multi-panel figures."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=fontsize, fontweight="bold", color=INK)


def model_color(run_name: str) -> str:
    """Accent for the proposed model, muted slate for every baseline."""
    if "custom_efficientnet_v2" in run_name:
        return PROPOSED_COLOR
    return BASELINE_COLOR


def is_proposed(run_name: str) -> bool:
    return "custom_efficientnet_v2" in run_name


def short_label(run_name: str) -> str:
    return MODEL_LABELS.get(run_name, run_name)


def draw_proposed_star(ax, x, y, size=320, color=PROPOSED_COLOR, edge=PROPOSED_EDGE):
    """Star marker highlighting the proposed model on a scatter plot."""
    ax.scatter([x], [y], marker="*", s=size, c=color, edgecolors=edge,
               linewidths=1.1, zorder=12)


def legend_clean(ax, **kwargs):
    """A legend with the house style: no frame unless asked, tight spacing."""
    defaults = dict(frameon=False, fontsize=9.5, handlelength=1.5,
                    borderaxespad=0.4, labelspacing=0.45)
    defaults.update(kwargs)
    leg = ax.legend(**defaults)
    if leg and defaults.get("frameon"):
        leg.get_frame().set_edgecolor(HAIRLINE)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_facecolor("white")
    return leg
