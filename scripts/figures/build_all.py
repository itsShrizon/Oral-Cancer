"""Orchestrator: run every fig*.py and report what was produced.

Usage:
    python scripts/figures/build_all.py               # all figures
    python scripts/figures/build_all.py --only fig01  # substrings
    python scripts/figures/build_all.py --skip fig06c # substrings
    python scripts/figures/build_all.py --check       # verify outputs exist
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIG_ROOT = REPO_ROOT / "figures"


SCRIPTS = [
    # Architecture (9)
    "fig01_custom_arch.py",
    "fig01b_attentionhub_detail.py",
    "fig01c_hub_v1_vs_v2.py",
    "fig01d_attention_internals.py",
    "fig01e_baseline_arches.py",
    "fig01i_depth_param_overview.py",
    # Dataset (2)
    "fig02_dataset_grid.py",
    "fig02b_class_distribution.py",
    # Benchmark (5)
    "fig03_pareto_params.py",
    "fig03b_pareto_flops.py",
    "fig03c_efficiency_2x2.py",
    "fig03d_radar.py",
    "fig03e_table5_delta.py",
    # Per-class (3)
    "fig04_per_class_heatmap.py",
    "fig04b_per_class_proposed.py",
    "fig04c_confusion_proposed.py",
    # Ablation (4)
    "fig05_ablation_bars.py",
    "fig05b_complementarity.py",
    "fig05c_ablation_scatter.py",
    "fig05d_v1_v2_progression.py",
    # Explainability (3)
    "fig06_gradcam_composite.py",
    "fig06b_lime_composite.py",
    "fig06c_proposed_explain.py",
    "fig06c_proposed_explain_landscape.py",
    "fig06d_gradcam_vs_lime.py",
    # Latency (2)
    "fig07_latency_kde.py",
    "fig07b_latency_boxplot.py",
]


def filter_scripts(only: list[str], skip: list[str]) -> list[str]:
    out = []
    for s in SCRIPTS:
        if only and not any(o in s for o in only):
            continue
        if skip and any(k in s for k in skip):
            continue
        out.append(s)
    return out


def run_script(name: str) -> bool:
    print(f"\n=== {name} ===", flush=True)
    res = subprocess.run([sys.executable, str(HERE / name)],
                         cwd=str(REPO_ROOT))
    return res.returncode == 0


def check_outputs() -> int:
    failures = 0
    for sub in ["01_architecture", "02_dataset", "03_benchmark",
                "04_per_class", "05_ablation", "06_explainability", "07_latency"]:
        d = FIG_ROOT / sub
        files = sorted(d.glob("*.png")) if d.exists() else []
        print(f"\n{sub}: {len(files)} PNGs")
        for f in files:
            size_kb = f.stat().st_size // 1024
            mark = "OK" if size_kb > 5 else "WARN"
            print(f"  [{mark}] {f.name}  ({size_kb} KB)")
            if size_kb < 5:
                failures += 1
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        rc = check_outputs()
        print(f"\n{'-' * 60}\nVerification: "
              + ("OK" if rc == 0 else f"{rc} suspiciously small file(s)"))
        return

    scripts = filter_scripts(args.only, args.skip)
    print(f"Running {len(scripts)} figure scripts ...")
    fails = []
    for s in scripts:
        if not run_script(s):
            fails.append(s)
    print(f"\n{'-' * 60}")
    print(f"Done: {len(scripts) - len(fails)}/{len(scripts)} successful")
    if fails:
        print("Failed:")
        for f in fails:
            print(f"  - {f}")
    check_outputs()


if __name__ == "__main__":
    main()
