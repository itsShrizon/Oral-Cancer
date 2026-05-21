"""Quick build-only smoke test for every AttentionHub ablation variant.

Constructs each variant, runs a 1-image forward pass, prints param count.
Does NOT load any checkpoint and does NOT train.
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from utils.ablation import ABLATIONS, branches_for
from models.architecture import MultiTaskOralClassifier


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    x = torch.zeros(1, 3, 224, 224)
    print(f"{'variant':<15} {'params':>12}  {'feature dim':>12}")
    print("-" * 45)
    for key in ABLATIONS:
        branches = branches_for(key)
        m = MultiTaskOralClassifier(
            backbone="custom_efficientnet_v2",
            attention_branches=branches,
        )
        m.eval()
        with torch.no_grad():
            yb, ys = m(x)
        n = count_params(m)
        print(f"{key:<15} {n:>12,}  {yb.shape[-1]}+{ys.shape[-1]}")


if __name__ == "__main__":
    main()
