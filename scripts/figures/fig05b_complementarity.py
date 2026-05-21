"""fig05b — Role-complementarity 3x3 heatmap (BAM, Triplet, KAN)."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from _lib.style import apply_rc, save_fig, FAIL_RED, INK, MUTED, HAIRLINE
from _lib.data_loader import collect_ablation

CEILING = 98.4


def main():
    apply_rc()
    rows = {r["variant"]: r["subtype_acc"] * 100 for r in collect_ablation()}

    modules = ["BAM", "Triplet", "KAN"]
    roles = {"BAM": "mixed spatial + channel",
             "Triplet": "cross-dimensional spatial",
             "KAN": "B-spline channel"}
    pair_map = {
        ("BAM", "BAM"): rows.get("bam"),
        ("Triplet", "Triplet"): rows.get("triplet"),
        ("KAN", "KAN"): rows.get("kan"),
        ("BAM", "Triplet"): rows.get("bam_triplet"),
        ("Triplet", "BAM"): rows.get("bam_triplet"),
        ("BAM", "KAN"): rows.get("bam_kan"),
        ("KAN", "BAM"): rows.get("bam_kan"),
        ("Triplet", "KAN"): rows.get("triplet_kan"),
        ("KAN", "Triplet"): rows.get("triplet_kan"),
    }
    M = np.array([[pair_map.get((a, b), np.nan) for b in modules] for a in modules])

    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=98.0, vmax=99.6, aspect="equal")

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Subtype accuracy (%)", fontsize=10, color=MUTED)
    cb.outline.set_edgecolor(HAIRLINE)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(colors=MUTED, length=3)

    ax.set_xticks(range(3))
    ax.set_xticklabels(modules, fontsize=11.5, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(modules, fontsize=11.5, fontweight="bold")

    for i in range(3):
        for j in range(3):
            v = M[i, j]
            if np.isnan(v):
                continue
            failing = (i != j and v <= CEILING)
            ax.text(j, i - (0.10 if failing else 0.0), f"{v:.2f}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="#FFFFFF" if 98.55 < v < 99.0 else INK)
            if failing:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#7A1410", lw=2.6, zorder=5))
                ax.text(j, i + 0.26, "role conflict", ha="center", va="center",
                        fontsize=8.2, color="#7A1410", style="italic")

    ax.set_title("Role-complementarity matrix\ndiagonal = single module · off-diagonal = pair",
                 pad=12)
    ax.set_xlabel("Module B")
    ax.set_ylabel("Module A")
    role_text = "     ".join(f"{m} — {roles[m]}" for m in modules)
    fig.text(0.5, 0.035, role_text, ha="center", va="bottom", fontsize=9,
             color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, "05_ablation", "fig05b_role_complementarity_matrix", tight=False)


if __name__ == "__main__":
    main()
