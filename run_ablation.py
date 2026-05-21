"""
AttentionHub ablation runner.

Trains and evaluates each AttentionHub variant of Custom EfficientNet V2 under
the SAME fair training recipe used by the 8 pretrained baselines (lr=1e-4, no
warmup/AMP/clip, ES=15, no TTA). This keeps the ablation table directly
comparable to the main results table.

Variants (8 total):
    none         - donor EfficientNetV2-B0 Block-4 (no attention; canonical control)
    bam          - BAM only
    triplet      - Triplet Attention only
    kan          - KAN only
    bam_triplet  - BAM + Triplet
    bam_kan      - BAM + KAN
    triplet_kan  - Triplet + KAN
    full         - BAM + Triplet + KAN  (== proposed model; reuses existing
                                         baseline_recipe run, NOT retrained)

Outputs go to:
    results/custom_efficientnet_v2_ablation_<key>/
except 'full' which is read from results/custom_efficientnet_v2_baseline_recipe/.

Usage:
    python run_ablation.py                       # run all variants except 'full'
    python run_ablation.py --only bam triplet    # subset
    python run_ablation.py --skip-train          # only evaluate / metrics
    python run_ablation.py --force               # rerun even if outputs exist
"""

import argparse
import os
import subprocess
import sys

from utils.ablation import ABLATIONS, run_name_for


BACKBONE = "custom_efficientnet_v2"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 'full' is the existing proposed model — never retrained by this runner.
DEFAULT_VARIANTS = [k for k in ABLATIONS.keys() if k != "full"]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!!! Command failed (exit {r.returncode}): {' '.join(cmd)}")
    return r.returncode == 0


def has_output(folder, name):
    return os.path.exists(os.path.join(RESULTS_DIR, folder, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"Subset of variants to run. Choices: {DEFAULT_VARIANTS}")
    ap.add_argument("--skip-train",   action="store_true", help="Skip training step")
    ap.add_argument("--skip-eval",    action="store_true", help="Skip evaluation step")
    ap.add_argument("--skip-metrics", action="store_true", help="Skip performance metrics step")
    ap.add_argument("--force", action="store_true",
                    help="Re-run a step even if its output already exists")
    args = ap.parse_args()

    variants = args.only if args.only else DEFAULT_VARIANTS
    for v in variants:
        if v not in ABLATIONS:
            print(f"Unknown variant: {v!r}. Choices: {sorted(ABLATIONS.keys())}")
            sys.exit(1)
        if v == "full":
            print(f"Skipping 'full' — proposed model already at "
                  f"results/{BACKBONE}_baseline_recipe/")
            continue

    py = sys.executable

    for v in variants:
        if v == "full":
            continue
        run_name = run_name_for(BACKBONE, v, recipe="baseline")
        print(f"\n{'='*70}\n  Variant: {v}  ->  results/{run_name}/\n{'='*70}")

        # 1. Train
        if not args.skip_train:
            if args.force or not has_output(run_name, "best_model.pth"):
                ok = run([py, "train.py", "--backbone", BACKBONE, "--ablation", v])
                if not ok:
                    print(f"   train.py failed for {v}; continuing to next variant.")
                    continue
            else:
                print(f"   [skip train] best_model.pth exists at {run_name}/")

        # 2. Evaluate (test set)
        if not args.skip_eval:
            if args.force or not has_output(run_name, "evaluation_results.txt"):
                run([py, "evaluate_final.py", "--backbone", BACKBONE,
                     "--ablation", v, "--no-confirm"])
            else:
                print(f"   [skip eval] evaluation_results.txt exists at {run_name}/")

        # 3. Performance metrics (FLOPs / latency / size)
        if not args.skip_metrics:
            if args.force or not has_output(run_name, "performance_metrics.json"):
                run([py, "compute_model_metrics.py", "--backbone", BACKBONE,
                     "--ablation", v, "--skip-gradcam", "--skip-shap"])
            else:
                print(f"   [skip metrics] performance_metrics.json exists at {run_name}/")

    print("\nAll requested variants done.")


if __name__ == "__main__":
    main()
