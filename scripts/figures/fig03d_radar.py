"""fig03d — Radar chart comparing proposed Custom V2 (Hub v2) vs strongest baselines."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import apply_rc, save_fig, PROPOSED_COLOR, INK, MUTED, HAIRLINE
from _lib.data_loader import collect_table1, PROPOSED_V2

SERIES_COLORS = ["#0072B2", "#009E73", PROPOSED_COLOR]


def main():
    apply_rc()
    rows = {r["run"]: r for r in collect_table1()}
    keys = ["efficientnet_v2b2", "inception_v3", PROPOSED_V2]
    runs = [rows[k] for k in keys]

    metric_names = ["Binary\naccuracy", "Subtype\naccuracy", "Parameter\nefficiency",
                    "FLOPs\nefficiency", "Size\nefficiency", "Memory\nefficiency"]
    raw = []
    for r in runs:
        raw.append([
            r["binary_acc"] * 100,
            r["subtype_acc"] * 100,
            1 / max(r["params_m"], 0.01),
            1 / max(r["gflops"], 0.001),
            1 / max(r["size_mb"], 0.01),
            1 / max(r["gpu_peak_mb"], 0.01),
        ])
    raw = np.array(raw)
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        lo, hi = raw[:, j].min(), raw[:, j].max()
        norm[:, j] = 1.0 if hi - lo < 1e-9 else 0.4 + 0.6 * (raw[:, j] - lo) / (hi - lo)

    n = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.8, 8.0), subplot_kw=dict(polar=True))
    lws = [2.0, 2.0, 2.8]
    fills = [0.10, 0.10, 0.20]
    for idx in (0, 1, 2):
        vals = norm[idx].tolist() + [norm[idx][0]]
        z = 6 if idx == 2 else 3
        ax.plot(angles, vals, "-", lw=lws[idx], color=SERIES_COLORS[idx],
                marker="o", markersize=6 if idx == 2 else 4.5, zorder=z,
                label=runs[idx]["label"])
        ax.fill(angles, vals, color=SERIES_COLORS[idx], alpha=fills[idx], zorder=z - 0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9.5, color=INK)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.08)
    ax.set_rlabel_position(0)
    ax.tick_params(colors=MUTED)
    ax.spines["polar"].set_color(HAIRLINE)
    ax.spines["polar"].set_linewidth(0.8)
    ax.grid(color="#DCDCDC", lw=0.7, linestyle=":")
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              frameon=False, fontsize=9.7)
    fig.suptitle("Proposed model vs. two strongest baselines  (6-axis comparison)",
                 x=0.015, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.045,
             "Efficiency axes are inverted (1 / metric) so that further from the centre is always better.",
             ha="center", fontsize=8.6, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(fig, "03_benchmark", "fig03d_proposed_vs_baselines_radar", tight=False)


if __name__ == "__main__":
    main()
