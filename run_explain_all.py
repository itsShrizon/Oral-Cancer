"""
Run explain_model.py on every model that has a trained checkpoint.

Usage:
    python run_explain_all.py
    python run_explain_all.py --skip-lime         # GradCAM only (much faster)
    python run_explain_all.py --models vgg19 resnet50
"""
import os
import sys
import subprocess
import argparse

ALL_MODELS = [
    'resnet50', 'densenet121', 'convnext_tiny', 'swin_t',
    'efficientnet_b0', 'efficientnet_v2b2', 'efficientnet_v2b3',
    'efficientnet_v2s', 'vgg19', 'inception_v3',
    'custom_efficientnet_v2',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=ALL_MODELS)
    parser.add_argument('--skip-lime', action='store_true')
    parser.add_argument('--skip-gradcam', action='store_true')
    args = parser.parse_args()

    for m in args.models:
        ckpt = os.path.join('results', m, 'best_model.pth')
        if not os.path.exists(ckpt):
            print(f"[skip] {m} — no checkpoint at {ckpt}")
            continue

        print(f"\n{'='*60}\n  EXPLAIN: {m}\n{'='*60}")
        cmd = [sys.executable, 'explain_model.py', '--backbone', m]
        if args.skip_lime:    cmd.append('--skip-lime')
        if args.skip_gradcam: cmd.append('--skip-gradcam')

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED ({e.returncode}) — continuing")


if __name__ == "__main__":
    main()
