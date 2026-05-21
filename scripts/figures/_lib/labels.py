"""Greedy label de-collision with leader lines for scatter / Pareto plots.

`smart_annotate` places point labels at the closest clear slot on a ring of
candidate offsets, scoring each candidate by overlap with already-placed
labels and with every data marker. Labels pushed far from their anchor get a
thin leader line. Final labels are emitted in data coordinates so the layout
survives a later tight_layout / bbox='tight' save.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from matplotlib.transforms import IdentityTransform


def _overlap(a, b, pad: float = 2.0) -> float:
    """Overlap area (display px) of two bboxes given as (x0, y0, x1, y1)."""
    dx = min(a[2], b[2]) - max(a[0], b[0]) + 2 * pad
    dy = min(a[3], b[3]) - max(a[1], b[1]) + 2 * pad
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def smart_annotate(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    fontsize=8.6,
    colors: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[str]] = None,
    leader_color: str = "#AEAEAE",
    leader_lw: float = 0.6,
    min_leader_px: float = 21.0,
    prefer: str = "up",
    avoid_pad: float = 3.0,
    point_radii: Optional[Sequence[float]] = None,
):
    """Place non-overlapping labels next to (xs, ys) data points.

    prefer: 'up' biases candidate slots above the marker, 'down' below.
    point_radii: per-point marker radius in display px, so labels clear large
        markers (e.g. a big star). Defaults to a small uniform radius.
    Returns the list of created annotation artists (anchor order).
    """
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return []

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    anchors = ax.transData.transform(np.column_stack([xs, ys]))
    axes_bb = ax.get_window_extent(renderer=renderer)
    if point_radii is None:
        point_radii = [8.0] * n

    base_ang = [90, 60, 120, 45, 135, 30, 150, 18, 162, 0, 180,
                -30, -45, -60, -90, -120, -135, -150]
    if prefer == "down":
        base_ang = [-a for a in base_ang]
    candidates = []
    for radius in (18, 26, 35, 46, 60, 78, 100, 126):
        for ang in base_ang:
            rad = math.radians(ang)
            candidates.append((radius * math.cos(rad),
                               radius * math.sin(rad), radius))

    probe = ax.text(0, 0, "", transform=IdentityTransform(),
                    ha="center", va="center", zorder=40)

    placed_boxes = []
    results = []  # (idx, text_display_xy, radius)
    order = sorted(range(n), key=lambda i: -anchors[i][1])  # top-to-bottom

    for i in order:
        ax_x, ax_y = anchors[i]
        fs = fontsize[i] if isinstance(fontsize, (list, tuple, np.ndarray)) else fontsize
        wt = weights[i] if weights else "normal"
        probe.set_text(labels[i])
        probe.set_fontsize(fs)
        probe.set_fontweight(wt)

        best, best_score = None, float("inf")
        for dx, dy, radius in candidates:
            cx, cy = ax_x + dx, ax_y + dy
            probe.set_position((cx, cy))
            bb = probe.get_window_extent(renderer=renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            score = 0.010 * radius * radius  # prefer the closest clear slot
            for pb in placed_boxes:
                score += _overlap(box, pb) * 16.0
            for k, (px, py) in enumerate(anchors):
                # distance from the marker centre to the nearest point of the box
                nx = min(max(px, box[0]), box[2])
                ny = min(max(py, box[1]), box[3])
                dist = math.hypot(px - nx, py - ny)
                clear = point_radii[k] + avoid_pad
                if dist < clear:
                    score += (clear - dist) * 45.0
            if (box[0] < axes_bb.x0 or box[2] > axes_bb.x1 or
                    box[1] < axes_bb.y0 or box[3] > axes_bb.y1):
                score += 220.0
            if score < best_score:
                best_score, best = score, (cx, cy, box, radius)
        cx, cy, box, radius = best
        placed_boxes.append(box)
        results.append((i, (cx, cy), radius))

    probe.remove()

    inv = ax.transData.inverted()
    out = [None] * n
    for i, text_disp, radius in results:
        fs = fontsize[i] if isinstance(fontsize, (list, tuple, np.ndarray)) else fontsize
        col = colors[i] if colors else "#333333"
        wt = weights[i] if weights else "normal"
        text_data = inv.transform(text_disp)
        arrowprops = None
        if radius >= min_leader_px:
            arrowprops = dict(arrowstyle="-", color=leader_color, lw=leader_lw,
                              shrinkA=1.0, shrinkB=2.5, connectionstyle="arc3")
        out[i] = ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=text_data,
                             textcoords="data", ha="center", va="center",
                             fontsize=fs, color=col, fontweight=wt, zorder=30,
                             arrowprops=arrowprops)
    return out
