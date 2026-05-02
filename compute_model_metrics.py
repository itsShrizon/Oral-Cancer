"""
Comprehensive performance metrics for trained oral cancer models.

Metrics computed and saved to results/{backbone}/:
  - performance_metrics.json   : FLOPs, timing, size, memory, energy, carbon, latency stats
  - latency_distribution.png   : histogram + CDF of single-image inference latency
  - gradcam_binary.png         : GradCAM++ on binary head (Benign/Malignant)
  - gradcam_subtype.png        : GradCAM++ on subtype head (7 classes)
  - shap_binary.png            : SHAP gradient explanation for binary head
  - shap_subtype.png           : SHAP gradient explanation for subtype head

Usage:
    python compute_model_metrics.py --backbone resnet50
    python compute_model_metrics.py --backbone custom_efficientnet_v2 --skip-shap

Optional dependencies (install as needed):
    pip install thop            # FLOPs
    pip install grad-cam        # GradCAM++
    pip install shap            # SHAP values
    pip install codecarbon      # Energy / carbon tracking
    pip install psutil          # CPU memory
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')

# ?? Optional library imports ????????????????????????????????????????????????

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Note: 'thop' not found - FLOPs skipped.  pip install thop")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("Note: 'codecarbon' not found - energy/carbon will be estimated.  pip install codecarbon")

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("Note: 'pytorch-grad-cam' not found - GradCAM skipped.  pip install grad-cam")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Note: 'shap' not found - SHAP skipped.  pip install shap")

# ?? Project imports ?????????????????????????????????????????????????????????

from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT, DS2_CLASSES
)
from configs import config
from utils.common import set_seed, get_device
from data.transforms import val_transform
from data.dataset import OralPathologyDataset, load_dataset1_split, load_dataset2_split
from models.architecture import MultiTaskOralClassifier

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

# ============================================================
# Single-head wrappers (GradCAM / SHAP need one output tensor)
# ============================================================

class BinaryHeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out_b, _ = self.model(x)
        return out_b


class SubtypeHeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, out_s = self.model(x)
        return out_s


# ============================================================
# Target layer resolution for GradCAM
# ============================================================

def get_gradcam_target_layer(model, backbone_name):
    """
    Returns (target_layers, reshape_transform).
    reshape_transform is None for CNN backbones; required for Swin Transformer.
    """
    backbone = model.backbone
    try:
        if backbone_name == 'custom_efficientnet_v2':
            return [backbone.stage5], None

        elif backbone_name == 'resnet50':
            return [backbone.layer4[-1]], None

        elif backbone_name == 'densenet121':
            return [backbone.features.denseblock4], None

        elif backbone_name == 'convnext_tiny':
            return [backbone.stages[-1].blocks[-1]], None

        elif backbone_name == 'swin_t':
            # Swin outputs (B, H*W, C) - need to reshape to (B, C, H, W) for GradCAM
            def swin_reshape(tensor, height=7, width=7):
                result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
                return result.permute(0, 3, 1, 2)
            return [backbone.layers[-1].blocks[-1].norm1], swin_reshape

        elif backbone_name in ('efficientnet_b0', 'efficientnet_v2b2',
                               'efficientnet_v2b3', 'efficientnet_v2s'):
            return [backbone.blocks[-1]], None

        elif backbone_name == 'vgg19':
            return [backbone.features[-1]], None

        elif backbone_name == 'inception_v3':
            return [backbone.Mixed_7c], None

        else:
            print(f"  No GradCAM target layer defined for '{backbone_name}'")
            return None, None

    except Exception as exc:
        print(f"  GradCAM target layer lookup failed: {exc}")
        return None, None


# ============================================================
# Metric helpers
# ============================================================

def compute_flops(model, device):
    """GFLOPs and parameter count via thop (wraps multi-output model)."""
    if not HAS_THOP:
        return None, None
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): b, _ = self.m(x); return b

    try:
        flops, params = thop_profile(_Wrapper(model), inputs=(dummy,), verbose=False)
        return int(flops), int(params)
    except Exception as exc:
        print(f"  FLOPs failed: {exc}")
        return None, None


def compute_timing(model, test_loader, device, n_warmup=5, n_batch=20, n_single=100):
    """
    Returns:
      batch_time_ms        - average forward-pass time over a full batch
      estimated_epoch_s    - batch_time * num_batches_per_epoch
      inference_mean_ms    - average single-image forward-pass time
      inference_std_ms     - std of single-image times
      single_times_ms      - raw list (used for latency distribution)
    """
    model.eval()

    sample_img = next(iter(test_loader))[0][0:1].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_img)

    # Single-image timing
    single_times = []
    with torch.no_grad():
        for _ in range(n_single):
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(sample_img)
            if device.type == 'cuda': torch.cuda.synchronize()
            single_times.append((time.perf_counter() - t0) * 1000)

    # Batch timing
    batch_times = []
    with torch.no_grad():
        for i, (imgs, _, _) in enumerate(test_loader):
            if i >= n_batch: break
            imgs = imgs.to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(imgs)
            if device.type == 'cuda': torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) * 1000)

    avg_batch = float(np.mean(batch_times))
    est_epoch = avg_batch * len(test_loader) / 1000  # seconds

    return {
        'batch_time_ms':       round(avg_batch, 3),
        'estimated_epoch_s':   round(est_epoch, 3),
        'inference_mean_ms':   round(float(np.mean(single_times)), 3),
        'inference_std_ms':    round(float(np.std(single_times)),  3),
        'single_times_ms':     single_times,
    }


def compute_model_size_mb(model_path):
    if not os.path.exists(model_path):
        return None
    return round(os.path.getsize(model_path) / (1024 ** 2), 3)


def compute_memory(model, device, batch_size=4):
    """Peak GPU memory (if CUDA) and process RSS (if psutil available)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224).to(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    cpu_before = 0
    if HAS_PSUTIL:
        cpu_before = psutil.Process().memory_info().rss

    with torch.no_grad():
        model(dummy)

    result = {}
    if device.type == 'cuda':
        result['gpu_peak_mb'] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    else:
        result['gpu_peak_mb'] = None

    if HAS_PSUTIL:
        cpu_after = psutil.Process().memory_info().rss
        result['cpu_rss_mb'] = round(cpu_after / 1024**2, 2)
    else:
        result['cpu_rss_mb'] = None

    return result


def compute_energy_carbon(model, test_loader, device, backbone_name, save_dir, n_batches=10):
    """
    Energy (kWh) and carbon emissions (kg CO2) during inference.
    Uses codecarbon when available; otherwise provides a rough estimate.
    """
    model.eval()

    if HAS_CODECARBON:
        try:
            tracker = EmissionsTracker(
                project_name=f"oral_cancer_{backbone_name}",
                output_dir=save_dir,
                log_level='error',
                save_to_file=True,
            )
            tracker.start()
            with torch.no_grad():
                for i, (imgs, _, _) in enumerate(test_loader):
                    if i >= n_batches: break
                    model(imgs.to(device))
            emissions_kg = tracker.stop()
            energy_kwh = getattr(getattr(tracker, 'final_emissions_data', None),
                                  'energy_consumed', None)
            return {
                'carbon_emission_kg_co2': round(float(emissions_kg), 8) if emissions_kg else None,
                'energy_kwh':             round(float(energy_kwh),   8) if energy_kwh   else None,
                'energy_source':          'codecarbon',
            }
        except Exception as exc:
            print(f"  CodeCarbon failed: {exc}")

    # Fallback: rough estimate from wall-clock time
    with torch.no_grad():
        t0 = time.perf_counter()
        for i, (imgs, _, _) in enumerate(test_loader):
            if i >= n_batches: break
            model(imgs.to(device))
        elapsed_s = time.perf_counter() - t0

    power_w   = 50 if device.type == 'cuda' else 15          # rough
    energy_kwh = power_w * elapsed_s / 3_600_000
    carbon_kg  = energy_kwh * 0.233                           # global avg ~0.233 kg CO2/kWh
    return {
        'carbon_emission_kg_co2': round(carbon_kg, 8),
        'energy_kwh':             round(energy_kwh, 8),
        'energy_source':          'estimated (install codecarbon for accuracy)',
    }


def compute_latency_distribution(model, test_loader, device, n_samples, save_path):
    """
    Runs n_samples single-image inferences and saves a latency histogram + CDF.
    Returns percentile / summary statistics dict.
    """
    model.eval()
    images_pool = []
    for imgs, _, _ in test_loader:
        images_pool.extend(imgs)
        if len(images_pool) >= n_samples:
            break
    images_pool = images_pool[:n_samples]

    # Warmup
    dummy = images_pool[0].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy)

    latencies = []
    with torch.no_grad():
        for img in images_pool:
            inp = img.unsqueeze(0).to(device)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inp)
            if device.type == 'cuda': torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    stats = {
        'mean_ms': round(float(arr.mean()), 3),
        'std_ms':  round(float(arr.std()),  3),
        'min_ms':  round(float(arr.min()),  3),
        'max_ms':  round(float(arr.max()),  3),
        'p50_ms':  round(float(np.percentile(arr, 50)), 3),
        'p90_ms':  round(float(np.percentile(arr, 90)), 3),
        'p95_ms':  round(float(np.percentile(arr, 95)), 3),
        'p99_ms':  round(float(np.percentile(arr, 99)), 3),
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(arr, bins=30, color='steelblue', edgecolor='black', alpha=0.75)
    axes[0].axvline(stats['mean_ms'], color='red',    linestyle='--',
                    label=f"Mean {stats['mean_ms']:.1f}ms")
    axes[0].axvline(stats['p95_ms'],  color='orange', linestyle='--',
                    label=f"P95  {stats['p95_ms']:.1f}ms")
    axes[0].set_title('Inference Latency Distribution')
    axes[0].set_xlabel('Latency (ms)'); axes[0].set_ylabel('Count')
    axes[0].legend()

    sorted_arr = np.sort(arr)
    axes[1].plot(sorted_arr, np.linspace(0, 1, len(sorted_arr)), color='steelblue')
    axes[1].axhline(0.95, color='orange', linestyle='--', label=f"P95={stats['p95_ms']:.1f}ms")
    axes[1].set_title('Cumulative Distribution'); axes[1].set_xlabel('Latency (ms)')
    axes[1].set_ylabel('Cumulative Probability'); axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Latency distribution -> {save_path}")
    return stats


# ============================================================
# GradCAM
# ============================================================

def _collect_samples(test_loader, n):
    """Collect up to n (image, label_b, label_s) tuples from the loader."""
    imgs_l, lb_l, ls_l = [], [], []
    for imgs, lb, ls in test_loader:
        for i in range(imgs.size(0)):
            imgs_l.append(imgs[i])
            lb_l.append(lb[i].item())
            ls_l.append(ls[i].item())
            if len(imgs_l) >= n:
                return imgs_l, lb_l, ls_l
    return imgs_l, lb_l, ls_l


def _denorm(tensor_chw):
    img = tensor_chw.cpu().numpy().transpose(1, 2, 0)
    img = IMAGENET_STD * img + IMAGENET_MEAN
    return np.clip(img, 0, 1).astype(np.float32)


def compute_gradcam(model, backbone_name, test_loader, device, save_dir, n_images=8):
    if not HAS_GRADCAM:
        print("  GradCAM skipped (pytorch-grad-cam not installed)")
        return

    target_layers, reshape_transform = get_gradcam_target_layer(model, backbone_name)
    if target_layers is None:
        return

    imgs_l, lb_l, ls_l = _collect_samples(test_loader, n_images)
    n_show = min(len(imgs_l), 4)

    # ?? Binary head ????????????????????????????????????????????????????????
    try:
        wrapper_b = BinaryHeadWrapper(model)
        wrapper_b.eval()
        cam_b = GradCAMPlusPlus(model=wrapper_b, target_layers=target_layers,
                                 reshape_transform=reshape_transform)

        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        if n_show == 1: axes = axes.reshape(2, 1)

        for i in range(n_show):
            inp     = imgs_l[i].unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(lb_l[i])]
            gcam    = cam_b(input_tensor=inp, targets=targets)[0]
            img_np  = _denorm(imgs_l[i])
            cam_img = show_cam_on_image(img_np, gcam, use_rgb=True)

            label = 'Benign' if lb_l[i] == 0 else 'Malignant'
            axes[0, i].imshow(img_np);  axes[0, i].set_title(f"Original\n({label})"); axes[0, i].axis('off')
            axes[1, i].imshow(cam_img); axes[1, i].set_title(f"GradCAM++\n({label})"); axes[1, i].axis('off')

        plt.suptitle(f'GradCAM++ - Binary Head ({backbone_name})', fontsize=13)
        plt.tight_layout()
        out = os.path.join(save_dir, 'gradcam_binary.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  GradCAM (binary) -> {out}")
    except Exception as exc:
        print(f"  GradCAM binary failed: {exc}")

    # ?? Subtype head ???????????????????????????????????????????????????????
    try:
        valid = [(imgs_l[i], ls_l[i]) for i in range(len(imgs_l)) if ls_l[i] != -1]
        if not valid:
            print("  No subtype-labeled images for GradCAM subtype")
            return
        n_sub = min(len(valid), 4)

        wrapper_s = SubtypeHeadWrapper(model)
        wrapper_s.eval()
        cam_s = GradCAMPlusPlus(model=wrapper_s, target_layers=target_layers,
                                 reshape_transform=reshape_transform)

        fig, axes = plt.subplots(2, n_sub, figsize=(4 * n_sub, 8))
        if n_sub == 1: axes = axes.reshape(2, 1)

        for pi, (img_t, cls) in enumerate(valid[:n_sub]):
            inp     = img_t.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(cls)]
            gcam    = cam_s(input_tensor=inp, targets=targets)[0]
            img_np  = _denorm(img_t)
            cam_img = show_cam_on_image(img_np, gcam, use_rgb=True)

            label = DS2_CLASSES[cls] if 0 <= cls < len(DS2_CLASSES) else str(cls)
            axes[0, pi].imshow(img_np);  axes[0, pi].set_title(f"Original\n({label})"); axes[0, pi].axis('off')
            axes[1, pi].imshow(cam_img); axes[1, pi].set_title(f"GradCAM++\n({label})"); axes[1, pi].axis('off')

        plt.suptitle(f'GradCAM++ - Subtype Head ({backbone_name})', fontsize=13)
        plt.tight_layout()
        out = os.path.join(save_dir, 'gradcam_subtype.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  GradCAM (subtype) -> {out}")
    except Exception as exc:
        print(f"  GradCAM subtype failed: {exc}")


# ============================================================
# SHAP
# ============================================================

def compute_shap(model, test_loader, device, save_dir, backbone_name,
                 n_background=30, n_test=4):
    if not HAS_SHAP:
        print("  SHAP skipped (shap not installed)")
        return

    try:
        # Collect images
        pool = []
        for imgs, _, _ in test_loader:
            pool.extend(list(imgs))
            if len(pool) >= n_background + n_test: break

        background  = torch.stack(pool[:n_background]).to(device)
        test_images = torch.stack(pool[n_background:n_background + n_test]).to(device)

        # ?? Binary head ?????????????????????????????????????????????????????
        wrapper_b = BinaryHeadWrapper(model)
        wrapper_b.eval()
        explainer = shap.GradientExplainer(wrapper_b, background)
        shap_vals = explainer.shap_values(test_images)        # list[n_classes][B, C, H, W]

        # Use class-1 (Malignant) SHAP values; average over channels
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        sv_map = np.abs(sv).mean(axis=1)                      # (n_test, H, W)
        test_np = test_images.cpu().numpy()

        fig, axes = plt.subplots(2, n_test, figsize=(4 * n_test, 8))
        if n_test == 1: axes = axes.reshape(2, 1)
        for i in range(n_test):
            img_disp = np.clip(IMAGENET_STD * test_np[i].transpose(1, 2, 0) + IMAGENET_MEAN, 0, 1)
            axes[0, i].imshow(img_disp); axes[0, i].set_title('Original'); axes[0, i].axis('off')
            im = axes[1, i].imshow(sv_map[i], cmap='hot')
            axes[1, i].set_title('SHAP |values|\n(Malignant)'); axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        plt.suptitle(f'SHAP - Binary Head ({backbone_name})', fontsize=13)
        plt.tight_layout()
        out = os.path.join(save_dir, 'shap_binary.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  SHAP (binary) -> {out}")

        # ?? Subtype head ?????????????????????????????????????????????????????
        wrapper_s = SubtypeHeadWrapper(model)
        wrapper_s.eval()
        explainer_s = shap.GradientExplainer(wrapper_s, background)
        shap_vals_s = explainer_s.shap_values(test_images)   # list[n_subtypes][B, C, H, W]

        # Sum |SHAP| across all subtype classes to show "any subtype" importance
        if isinstance(shap_vals_s, list):
            sv_s = np.stack([np.abs(v) for v in shap_vals_s]).sum(axis=0).mean(axis=1)
        else:
            sv_s = np.abs(shap_vals_s).mean(axis=1)

        fig, axes = plt.subplots(2, n_test, figsize=(4 * n_test, 8))
        if n_test == 1: axes = axes.reshape(2, 1)
        for i in range(n_test):
            img_disp = np.clip(IMAGENET_STD * test_np[i].transpose(1, 2, 0) + IMAGENET_MEAN, 0, 1)
            axes[0, i].imshow(img_disp); axes[0, i].set_title('Original'); axes[0, i].axis('off')
            im = axes[1, i].imshow(sv_s[i], cmap='hot')
            axes[1, i].set_title('SHAP |values|\n(Subtype)'); axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        plt.suptitle(f'SHAP - Subtype Head ({backbone_name})', fontsize=13)
        plt.tight_layout()
        out = os.path.join(save_dir, 'shap_subtype.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  SHAP (subtype) -> {out}")

    except Exception as exc:
        import traceback
        print(f"  SHAP failed: {exc}")
        traceback.print_exc()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Compute comprehensive model metrics')
    parser.add_argument('--backbone',     type=str, default=BACKBONE, help='Backbone name')
    parser.add_argument('--recipe',       type=str, default='tuned',
                        choices=['tuned', 'baseline'],
                        help='Which trained checkpoint folder to read from. Must match '
                             'the recipe used during training.')
    parser.add_argument('--skip-gradcam', action='store_true', help='Skip GradCAM')
    parser.add_argument('--skip-shap',    action='store_true', help='Skip SHAP')
    args = parser.parse_args()

    backbone_name  = args.backbone
    recipe         = args.recipe
    is_custom_arch = (backbone_name == 'custom_efficientnet_v2')

    run_name = backbone_name
    if is_custom_arch and recipe == 'baseline':
        run_name = f"{backbone_name}_baseline_recipe"

    save_dir       = os.path.join(config.BASE_PATH, 'results', run_name)
    model_path     = os.path.join(save_dir, 'best_model.pth')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE METRICS - {run_name} (recipe={recipe})")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}\nRun training first.")
        sys.exit(1)

    set_seed()
    device = get_device()

    # Load test dataset
    print("\nLoading test dataset...")
    d1p, d1b, d1s = load_dataset1_split('test')
    d2p, d2b, d2s = load_dataset2_split('test')
    test_ds = OralPathologyDataset(d1p + d2p, d1b + d2b, d1s + d2s, transform=val_transform)
    test_loader = DataLoader(
        test_ds, batch_size=min(BATCH_SIZE, 32), shuffle=False,
        num_workers=min(NUM_WORKERS, 4), pin_memory=True
    )
    print(f"Test images: {len(test_ds)}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = MultiTaskOralClassifier(backbone=backbone_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics = {'backbone': backbone_name, 'recipe': recipe, 'run_name': run_name}

    # ?? 1. Model size ??????????????????????????????????????????????????????
    print("\n[1/8] Model size...")
    metrics['model_size_mb'] = compute_model_size_mb(model_path)
    print(f"      {metrics['model_size_mb']} MB")

    # ?? 2. FLOPs ???????????????????????????????????????????????????????????
    print("\n[2/8] FLOPs...")
    flops, params = compute_flops(model, device)
    metrics['flops_raw']   = flops
    metrics['flops_gflops']= round(flops / 1e9, 3) if flops else None
    metrics['num_parameters'] = params if params else sum(p.numel() for p in model.parameters())
    if flops:
        print(f"      {metrics['flops_gflops']} GFLOPs | {metrics['num_parameters']:,} params")

    # ?? 3. Timing ??????????????????????????????????????????????????????????
    print("\n[3/8] Timing...")
    timing = compute_timing(model, test_loader, device)
    metrics['batch_time_ms']        = timing['batch_time_ms']
    metrics['estimated_epoch_time_s'] = timing['estimated_epoch_s']
    metrics['inference_time_ms_mean'] = timing['inference_mean_ms']
    metrics['inference_time_ms_std']  = timing['inference_std_ms']
    print(f"      batch={timing['batch_time_ms']}ms | "
          f"infer={timing['inference_mean_ms']}?{timing['inference_std_ms']}ms | "
          f"epoch?{timing['estimated_epoch_s']}s")

    # ?? 4. Memory ??????????????????????????????????????????????????????????
    print("\n[4/8] Memory usage...")
    mem = compute_memory(model, device)
    metrics.update(mem)
    if mem.get('gpu_peak_mb'): print(f"      GPU peak: {mem['gpu_peak_mb']} MB")
    if mem.get('cpu_rss_mb'):  print(f"      CPU RSS:  {mem['cpu_rss_mb']} MB")

    # ?? 5. Energy & Carbon ?????????????????????????????????????????????????
    print("\n[5/8] Energy & carbon...")
    ec = compute_energy_carbon(model, test_loader, device, backbone_name, save_dir)
    metrics.update(ec)
    print(f"      energy={ec.get('energy_kwh')} kWh | "
          f"CO2={ec.get('carbon_emission_kg_co2')} kg  [{ec.get('energy_source')}]")

    # ?? 6. Latency distribution ????????????????????????????????????????????
    print("\n[6/8] Latency distribution...")
    n_lat = min(200, len(test_ds))
    lat_path = os.path.join(save_dir, 'latency_distribution.png')
    lat_stats = compute_latency_distribution(model, test_loader, device, n_lat, lat_path)
    metrics['latency_distribution'] = lat_stats
    print(f"      P50={lat_stats['p50_ms']}ms | P95={lat_stats['p95_ms']}ms | "
          f"P99={lat_stats['p99_ms']}ms")

    # ?? 7. GradCAM ?????????????????????????????????????????????????????????
    if not args.skip_gradcam:
        print("\n[7/8] GradCAM++...")
        compute_gradcam(model, backbone_name, test_loader, device, save_dir)
    else:
        print("\n[7/8] GradCAM skipped (--skip-gradcam)")

    # ?? 8. SHAP ????????????????????????????????????????????????????????????
    if not args.skip_shap:
        print("\n[8/8] SHAP...")
        compute_shap(model, test_loader, device, save_dir, backbone_name)
    else:
        print("\n[8/8] SHAP skipped (--skip-shap)")

    # ?? Save JSON ??????????????????????????????????????????????????????????
    def _serial(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return str(obj)

    out_json = os.path.join(save_dir, 'performance_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=_serial)
    print(f"\nOK Metrics saved -> {out_json}")

    # ?? Summary ????????????????????????????????????????????????????????????
    print(f"\n{'='*60}")
    print(f"  SUMMARY - {backbone_name}")
    print(f"{'='*60}")
    print(f"  Model size (pt):        {metrics.get('model_size_mb')} MB")
    print(f"  FLOPs:                  {metrics.get('flops_gflops')} GFLOPs")
    print(f"  Parameters:             {metrics.get('num_parameters'):,}")
    print(f"  Batch time:             {metrics.get('batch_time_ms')} ms")
    print(f"  Inference time (mean):  {metrics.get('inference_time_ms_mean')} ms")
    print(f"  Epoch time (est.):      {metrics.get('estimated_epoch_time_s')} s")
    print(f"  GPU memory (peak):      {metrics.get('gpu_peak_mb')} MB")
    print(f"  CPU memory (RSS):       {metrics.get('cpu_rss_mb')} MB")
    print(f"  Energy:                 {metrics.get('energy_kwh')} kWh")
    print(f"  Carbon emission:        {metrics.get('carbon_emission_kg_co2')} kg CO2")
    ld = metrics.get('latency_distribution', {})
    print(f"  Latency P50/P95/P99:    {ld.get('p50_ms')}/{ld.get('p95_ms')}/{ld.get('p99_ms')} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
