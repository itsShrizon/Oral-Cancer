"""Architecture-diagram layout helpers (matplotlib patches)."""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def add_block(
    ax,
    xy: Tuple[float, float],
    w: float,
    h: float,
    title: str,
    sub: Optional[str] = None,
    shape: Optional[str] = None,
    facecolor: str = "#E8EEF6",
    edgecolor: str = "#5F7FA8",
    title_color: str = "#1A1A1A",
    fontsize_title: int = 11,
    fontsize_sub: int = 9.5,
    fontsize_shape: int = 9,
    rounding: float = 0.45,
    lw: float = 1.1,
    title_weight: str = "bold",
    shape_inside: bool = True,
):
    """Draw a rounded block with optional sub-text and a tensor-shape annotation.

    Layout inside the box:
      [title]
      [sub]            (smaller, regular)
      [shape]          (italic monospace, even smaller)
    Vertical positions are computed so that there is no overlap.
    """
    x, y = xy
    box = mpatches.FancyBboxPatch(
        (x + 0.1, y + 0.1),
        max(w - 0.2, 0.5),
        max(h - 0.2, 0.5),
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(box)
    cx = x + w / 2

    # Vertical anchors
    if sub is None and shape is None:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize_title, color=title_color, fontweight=title_weight)
        return

    # Multi-line layout
    has_sub = sub is not None
    has_shape = shape is not None and shape_inside

    if has_sub and has_shape:
        ax.text(cx, y + h * 0.74, title, ha="center", va="center",
                fontsize=fontsize_title, color=title_color, fontweight=title_weight)
        ax.text(cx, y + h * 0.46, sub, ha="center", va="center",
                fontsize=fontsize_sub, color=title_color)
        ax.text(cx, y + h * 0.20, shape, ha="center", va="center",
                fontsize=fontsize_shape, color="#555555",
                style="italic", family="monospace")
    elif has_sub:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=fontsize_title, color=title_color, fontweight=title_weight)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center",
                fontsize=fontsize_sub, color=title_color)
    elif has_shape and shape_inside:
        ax.text(cx, y + h * 0.62, title, ha="center", va="center",
                fontsize=fontsize_title, color=title_color, fontweight=title_weight)
        ax.text(cx, y + h * 0.28, shape, ha="center", va="center",
                fontsize=fontsize_shape, color="#555555",
                style="italic", family="monospace")
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize_title, color=title_color, fontweight=title_weight)

    # External shape annotation (above-right) only when explicitly requested
    if shape and not shape_inside:
        ax.text(x + w, y + h + 0.4, shape, ha="right", va="bottom",
                fontsize=fontsize_shape, color="#555555", style="italic",
                family="monospace")


def add_arrow(
    ax,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str = "#5A6270",
    lw: float = 1.3,
    style: str = "-|>",
    mutation_scale: float = 16,
):
    arr = mpatches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)


def connect_h(ax, box_a, box_b, **kw):
    """Connect right edge of box_a to left edge of box_b horizontally.

    box = (x, y, w, h)
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    p1 = (xa + wa, ya + ha / 2)
    p2 = (xb, yb + hb / 2)
    add_arrow(ax, p1, p2, **kw)


def connect_v(ax, box_a, box_b, **kw):
    """Connect bottom edge of box_a to top edge of box_b vertically (top-to-bottom flow)."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    # ya is the *top* y if axes are inverted; here we treat (x,y) as bottom-left
    p1 = (xa + wa / 2, ya)
    p2 = (xb + wb / 2, yb + hb)
    add_arrow(ax, p1, p2, **kw)


def setup_canvas(figsize=(18, 9), xlim=(0, 100), ylim=(0, 50), bg="white"):
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def text_callout(ax, x, y, text, fontsize=10, color="#1A1A1A",
                 boxcolor="#FBF3E4", edgecolor="#D6A94B"):
    bbox = dict(boxstyle="round,pad=0.5", facecolor=boxcolor,
                edgecolor=edgecolor, linewidth=1.0)
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha="center", va="center", bbox=bbox)
