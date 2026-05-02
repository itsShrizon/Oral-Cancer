"""
Automated experiment runner.

For each model this script runs four steps:
  1. Train      - python train.py --backbone <model>
  2. Evaluate   - python evaluate_final.py --backbone <model> --no-confirm
  3. Visualize  - python visualize_predictions.py --backbone <model>
  4. Metrics    - python compute_model_metrics.py --backbone <model>

Each step is skipped automatically when its output already exists.
Use --force to delete existing outputs and rerun everything from scratch.

Usage:
    python run_all_models.py              # run missing steps only
    python run_all_models.py --force      # rerun EVERYTHING from scratch
    python run_all_models.py --force --models resnet50 densenet121  # specific models
"""

import subprocess
import sys
import os
import shutil
import argparse

# Always run from the project root so train.py / evaluate_final.py / etc.
# resolve correctly regardless of where this script was invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

ALL_MODELS = [
    'resnet50',
    'densenet121',
    'convnext_tiny',
    'swin_t',
    'efficientnet_b0',
    'efficientnet_v2b2',
    'efficientnet_v2b3',
    'efficientnet_v2s',
    'vgg19',
    'inception_v3',
    'custom_efficientnet_v2',
]


def run_command(command):
    """Run a subprocess command; return True on success."""
    print(f"  $ {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Command failed (exit {e.returncode})")
        return False


def clear_model_results(model):
    """Delete the results folder for a model to force a full rerun."""
    folder = os.path.join('results', model)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"  Cleared {folder}")
    os.makedirs(folder, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Run all model experiments')
    parser.add_argument('--force',  action='store_true',
                        help='Delete existing results and rerun everything from scratch')
    parser.add_argument('--models', nargs='+', default=None,
                        metavar='MODEL',
                        help=f'Subset of models to run (default: all). '
                             f'Choices: {ALL_MODELS}')
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS

    # Validate model names
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        print(f"Unknown model(s): {unknown}")
        print(f"Valid choices: {ALL_MODELS}")
        sys.exit(1)

    print("=" * 60)
    print("AUTOMATED EXPERIMENT RUNNER")
    print(f"Models : {', '.join(models)}")
    print(f"Force  : {args.force}")
    print("=" * 60)

    if args.force:
        print("\n--force: deleting existing results for selected models...")
        for model in models:
            clear_model_results(model)

    for model in models:
        print(f"\n\n{'='*50}")
        print(f"  MODEL: {model}")
        print(f"{'='*50}")

        results_dir  = os.path.join('results', model)
        model_path   = os.path.join(results_dir, 'best_model.pth')
        eval_file    = os.path.join(results_dir, 'evaluation_results.txt')
        viz_file     = os.path.join(results_dir, 'prediction_samples.png')
        metrics_file = os.path.join(results_dir, 'performance_metrics.json')
        os.makedirs(results_dir, exist_ok=True)

        # ?? Step 1: Train ????????????????????????????????????????????????
        if os.path.exists(model_path) or os.path.exists(eval_file):
            print(f"\n[1/4] Train - skipped (model already exists)")
        else:
            print(f"\n[1/4] Training {model}...")
            ok = run_command([sys.executable, 'train.py', '--backbone', model])
            if not ok:
                print(f"  Training failed - skipping remaining steps for {model}.")
                continue

        # ?? Step 2: Evaluate ?????????????????????????????????????????????
        if os.path.exists(eval_file):
            print(f"\n[2/4] Evaluate - skipped (results already exist)")
        else:
            print(f"\n[2/4] Evaluating {model}...")
            run_command([sys.executable, 'evaluate_final.py',
                         '--backbone', model, '--no-confirm'])

        # ?? Step 3: Visualize ????????????????????????????????????????????
        if os.path.exists(viz_file):
            print(f"\n[3/4] Visualize - skipped (already done)")
        else:
            print(f"\n[3/4] Visualizing {model}...")
            run_command([sys.executable, 'visualize_predictions.py', '--backbone', model])

        # ?? Step 4: Performance metrics ???????????????????????????????????
        if os.path.exists(metrics_file):
            print(f"\n[4/4] Metrics - skipped (already computed)")
        else:
            print(f"\n[4/4] Computing metrics for {model}...")
            run_command([sys.executable, 'compute_model_metrics.py', '--backbone', model])

        print(f"\n  OK Completed {model}")

    print("\n\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
