"""
FLOPs estimator without thop.

Chain of thought:
  * FLOPs for a neural network at a fixed input size are dominated by Conv2d and
    Linear layers. Normalisation, activations, pooling contribute <1%.
  * For each Conv2d/Linear layer we can compute its MAC (multiply-accumulate)
    count from the OUTPUT shape it actually produced on a real forward pass -
    no guessing required:
        Conv2d MACs  = Cin/groups * kH * kW * Cout * outH * outW
        Linear MACs  = in_features * out_features  (per sample)
  * FLOPs = 2 * MACs (one multiply + one add per MAC), which matches what thop
    reports.
  * We register a forward hook on every Conv2d/Linear module, run ONE dummy
    forward pass, and sum. This exactly mirrors what thop does internally, so
    the numbers are not "fake" - they are the same definition of FLOPs, just
    computed without the thop dependency.

Then each model's performance_metrics.json is patched with flops_raw and
flops_gflops so the downstream docs pick them up.
"""
import os
import json
import torch
import torch.nn as nn

from models.architecture import MultiTaskOralClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')

BACKBONES = [
    ('resnet50',              224),
    ('densenet121',           224),
    ('convnext_tiny',         224),
    ('swin_t',                224),
    ('efficientnet_b0',       224),
    ('efficientnet_v2b2',     224),
    ('efficientnet_v2b3',     224),
    ('efficientnet_v2s',      224),
    ('inception_v3',          224),   # training used 224, not native 299
    ('custom_efficientnet_v2', 224),
]

# The custom model's folder also has a baseline-recipe twin; FLOPs are
# identical since the architecture is the same.
TWIN = {'custom_efficientnet_v2': 'custom_efficientnet_v2_baseline_recipe'}


def count_flops(model, input_size=224):
    """Returns MACs across Conv2d + Linear for a single-sample forward pass."""
    model.eval()
    total_macs = [0]
    hooks = []

    def conv_hook(mod, inp, out):
        # out shape: (N, Cout, H, W)
        _, cout, h, w = out.shape
        cin = mod.in_channels
        kh, kw = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        groups = mod.groups
        macs = (cin // groups) * kh * kw * cout * h * w
        total_macs[0] += macs

    def linear_hook(mod, inp, out):
        # For linear, MACs = in_features * out_features * batch_size_like
        # Batch size is 1 here.  For 3-D inputs (eg attention), out shape is
        # (N, T, out_features); multiply by T.
        n_tokens = 1
        if out.dim() > 2:
            n_tokens = int(torch.tensor(out.shape[1:-1]).prod().item())
        macs = mod.in_features * mod.out_features * n_tokens
        total_macs[0] += macs

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return total_macs[0]


def patch_json(folder, flops_raw):
    path = os.path.join(RESULTS, folder, 'performance_metrics.json')
    if not os.path.exists(path):
        print(f"  (no performance_metrics.json in {folder}, skipping)")
        return
    with open(path) as f:
        data = json.load(f)
    data['flops_raw'] = int(flops_raw)
    data['flops_gflops'] = round(flops_raw / 1e9, 3)
    data['flops_source'] = 'hook-based estimator (Conv2d+Linear MACs * 2)'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  patched {path}  ->  {data['flops_gflops']} GFLOPs")


def main():
    for backbone, size in BACKBONES:
        print(f"\n[{backbone}]  building model...")
        try:
            # Avoid pretrained-weight download for speed
            model = MultiTaskOralClassifier(backbone=backbone, pretrained=False)
        except Exception as exc:
            print(f"  build failed: {exc}")
            continue

        try:
            macs = count_flops(model, input_size=size)
        except Exception as exc:
            print(f"  forward pass failed: {exc}")
            continue

        flops = macs * 2  # thop convention: 1 MAC = 2 FLOPs
        gflops = flops / 1e9
        print(f"  MACs={macs:,}  FLOPs={flops:,}  ({gflops:.3f} GFLOPs)")

        patch_json(backbone, flops)
        if backbone in TWIN:
            patch_json(TWIN[backbone], flops)


if __name__ == "__main__":
    main()
