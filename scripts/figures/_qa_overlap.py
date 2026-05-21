"""QA: render every non-explainability figure and report problems.

Detects overlapping text labels and labels that spill outside the figure —
a programmatic stand-in for a visual pass. Run from the repo root:
    python scripts/figures/_qa_overlap.py
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
from matplotlib.text import Text, Annotation

import _lib.style as style


def _ascii(s):
    return s.encode("ascii", "replace").decode("ascii")

SCRIPTS = [
    "fig01_custom_arch", "fig01b_attentionhub_detail", "fig01c_hub_v1_vs_v2",
    "fig01d_attention_internals", "fig01e_baseline_arches", "fig01i_depth_param_overview",
    "fig02_dataset_grid", "fig02b_class_distribution",
    "fig03_pareto_params", "fig03b_pareto_flops", "fig03c_efficiency_2x2",
    "fig03d_radar", "fig03e_table5_delta",
    "fig04_per_class_heatmap", "fig04b_per_class_proposed", "fig04c_confusion_proposed",
    "fig05_ablation_bars", "fig05b_complementarity", "fig05c_ablation_scatter",
    "fig05d_v1_v2_progression",
    "fig07_latency_kde", "fig07b_latency_boxplot",
]

_real_save = style.save_fig
REPORT = []


def _ov_area(a, b):
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _checking_save(fig, category, name, **kw):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_bb = fig.bbox

    items = []
    for ax in fig.get_axes():
        for t in list(ax.texts):
            if t.get_text().strip():
                items.append(t)
    for t in list(fig.texts):
        if t.get_text().strip():
            items.append(t)

    boxes = []
    for t in items:
        try:
            # for annotations measure the TEXT only, not the leader line
            if isinstance(t, Annotation):
                bb = Text.get_window_extent(t, renderer=rend)
            else:
                bb = t.get_window_extent(renderer=rend)
        except Exception:
            continue
        boxes.append((_ascii(t.get_text().replace("\n", " ")[:26]), bb))

    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ov = _ov_area(boxes[i][1], boxes[j][1])
            if ov <= 1.0:
                continue
            a, b = boxes[i][1], boxes[j][1]
            amin = max(min(a.width * a.height, b.width * b.height), 1.0)
            frac = ov / amin
            if frac > 0.22:
                overlaps.append((boxes[i][0], boxes[j][0], frac))

    spills = []
    for txt, bb in boxes:
        if (bb.x0 < fig_bb.x0 - 2 or bb.x1 > fig_bb.x1 + 2 or
                bb.y0 < fig_bb.y0 - 2 or bb.y1 > fig_bb.y1 + 2):
            spills.append(txt)

    REPORT.append((name, len(boxes), overlaps, spills))
    return _real_save(fig, category, name, **kw)


style.save_fig = _checking_save


def main():
    failed = []
    for mod_name in SCRIPTS:
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
        except Exception:
            failed.append(mod_name)
            print(f"!! {mod_name} raised:")
            traceback.print_exc()

    print("\n" + "=" * 66)
    print("QA REPORT — text overlaps & spills")
    print("=" * 66)
    clean = 0
    for name, n_txt, overlaps, spills in REPORT:
        if not overlaps and not spills:
            clean += 1
            print(f"[ ok ]   {name}  ({n_txt} labels)")
            continue
        print(f"[FLAG]   {name}  ({n_txt} labels)")
        for a, b, frac in overlaps:
            print(f"           overlap {frac * 100:3.0f}%:  '{a}'  x  '{b}'")
        for s in spills:
            print(f"           spill outside figure:  '{s}'")
    print("-" * 66)
    print(f"{clean}/{len(REPORT)} figures clean; "
          f"{len(REPORT) - clean} flagged; {len(failed)} crashed")
    if failed:
        print("crashed:", ", ".join(failed))


if __name__ == "__main__":
    main()
