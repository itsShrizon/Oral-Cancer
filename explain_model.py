"""
Explainability for trained oral-cancer models.

Produces side-by-side LIME + GradCAM++ visualizations for both the binary
(Benign/Malignant) and subtype (7-class) heads. Use this to verify that the
network is attending to lesion regions rather than background artifacts.

Outputs (results/<backbone>/):
  - explain_binary.png    panel: Original | GradCAM++ | LIME (per sample)
  - explain_subtype.png   same layout for the 7-class head

Usage:
  python explain_model.py --backbone resnet50
  python explain_model.py --backbone vgg19 --num-samples 6 --skip-lime

Requires:
  pip install grad-cam lime scikit-image
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

from configs.config import BACKBONE, BATCH_SIZE, NUM_WORKERS, DS2_CLASSES
from configs import config
from utils.common import set_seed, get_device
from data.transforms import val_transform
from data.dataset import OralPathologyDataset, load_dataset1_split, load_dataset2_split
from models.architecture import MultiTaskOralClassifier
from compute_model_metrics import (
    BinaryHeadWrapper, SubtypeHeadWrapper, get_gradcam_target_layer,
)
from utils.ablation import ABLATIONS, branches_for, run_name_for

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm(tensor_chw):
    img = tensor_chw.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(IMAGENET_STD * img + IMAGENET_MEAN, 0, 1).astype(np.float32)


def collect_diverse_samples(test_loader, n_per_class, num_classes, label_idx):
    """Pick up to n_per_class samples for each label of head `label_idx` (1=binary, 2=subtype)."""
    buckets = {c: [] for c in range(num_classes)}
    for imgs, lb, ls in test_loader:
        labels = lb if label_idx == 1 else ls
        for i in range(imgs.size(0)):
            c = labels[i].item()
            if c == -1 or c not in buckets:
                continue
            if len(buckets[c]) < n_per_class:
                buckets[c].append((imgs[i], lb[i].item(), ls[i].item()))
        if all(len(v) >= n_per_class for v in buckets.values()):
            break
    out = []
    for c in range(num_classes):
        out.extend(buckets[c])
    return out


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def make_lime_predict_fn(model, device, head):
    """
    LIME calls this with a numpy batch (N, H, W, 3) in [0,1].
    We must return a (N, num_classes) numpy array of probabilities.
    """
    def predict(images_np):
        x = torch.from_numpy(images_np).float().permute(0, 3, 1, 2)  # NCHW
        x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = x.to(device)
        with torch.no_grad():
            out_b, out_s = model(x)
        logits = out_b if head == 'binary' else out_s
        return F.softmax(logits, dim=1).cpu().numpy()
    return predict


def explain_with_lime(predict_fn, img_rgb, target_label, num_samples=500):
    """Returns a (H, W, 3) overlay image highlighting top regions for target_label."""
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_rgb.astype(np.double),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42,
    )
    temp, mask = explanation.get_image_and_mask(
        target_label,
        positive_only=True,
        num_features=8,
        hide_rest=False,
    )
    return mark_boundaries(temp / 255.0 if temp.max() > 1.5 else temp, mask)


# ---------------------------------------------------------------------------
# Per-head explanation panel
# ---------------------------------------------------------------------------

def build_panel(model, backbone_name, samples, device, head, save_path,
                lime_samples=500, skip_gradcam=False, skip_lime=False):
    """
    samples: list of (img_tensor, binary_label, subtype_label)
    head:    'binary' or 'subtype'
    """
    n = len(samples)
    if n == 0:
        print(f"  [{head}] no samples available")
        return

    cols = 1 + (0 if skip_gradcam else 1) + (0 if skip_lime else 1)
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
    if n == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(-1, 1)

    # GradCAM setup
    cam = None
    if not skip_gradcam:
        target_layers, reshape = get_gradcam_target_layer(model, backbone_name)
        if target_layers is None:
            print(f"  GradCAM target layer not defined for {backbone_name}")
            skip_gradcam = True
        else:
            wrapper = BinaryHeadWrapper(model) if head == 'binary' else SubtypeHeadWrapper(model)
            wrapper.eval()
            cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers,
                                  reshape_transform=reshape)

    # LIME setup
    lime_predict = None
    if not skip_lime:
        lime_predict = make_lime_predict_fn(model, device, head)

    label_names_binary = ['Benign', 'Malignant']

    for row, (img_t, lb, ls) in enumerate(samples):
        target = lb if head == 'binary' else ls
        if head == 'binary':
            label_str = label_names_binary[target]
        else:
            label_str = DS2_CLASSES[target] if 0 <= target < len(DS2_CLASSES) else str(target)

        img_np = denorm(img_t)

        col = 0
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f"Original\n({label_str})")
        axes[row, col].axis('off')
        col += 1

        if not skip_gradcam:
            inp = img_t.unsqueeze(0).to(device)
            try:
                grayscale = cam(input_tensor=inp,
                                 targets=[ClassifierOutputTarget(target)])[0]
                cam_img = show_cam_on_image(img_np, grayscale, use_rgb=True)
                axes[row, col].imshow(cam_img)
                axes[row, col].set_title(f"GradCAM++\n(class {label_str})")
            except Exception as exc:
                axes[row, col].text(0.5, 0.5, f"GradCAM fail\n{exc}",
                                     ha='center', va='center',
                                     transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
            col += 1

        if not skip_lime:
            try:
                lime_img = explain_with_lime(lime_predict, img_np, target,
                                             num_samples=lime_samples)
                axes[row, col].imshow(lime_img)
                axes[row, col].set_title(f"LIME\n(class {label_str})")
            except Exception as exc:
                axes[row, col].text(0.5, 0.5, f"LIME fail\n{exc}",
                                     ha='center', va='center',
                                     transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

    plt.suptitle(f"Explainability — {head} head ({backbone_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_run_folder(folder_name):
    """Folder name -> (backbone, ablation, recipe, hub_version). None if unrecognised."""
    if folder_name == 'custom_efficientnet_v2_hub_v2':
        return 'custom_efficientnet_v2', None, 'baseline', 'v2'
    if folder_name == 'custom_efficientnet_v2_baseline_recipe':
        return 'custom_efficientnet_v2', 'full', 'baseline', 'v1'
    if folder_name.startswith('custom_efficientnet_v2_ablation_'):
        key = folder_name[len('custom_efficientnet_v2_ablation_'):]
        if key in ABLATIONS:
            return 'custom_efficientnet_v2', key, 'baseline', 'v1'
        return None
    return folder_name, None, 'tuned', 'v1'


def _explain_run(backbone, ablation, recipe, device, test_loader, args, hub_version='v1'):
    save_dir = os.path.join(config.BASE_PATH, 'results',
                             run_name_for(backbone, ablation, recipe,
                                          hub_version=hub_version))
    model_path = os.path.join(save_dir, 'best_model.pth')
    bin_out = os.path.join(save_dir, 'explain_binary.png')
    sub_out = os.path.join(save_dir, 'explain_subtype.png')

    if not os.path.exists(model_path):
        print(f"[skip] {save_dir} - no best_model.pth")
        return
    if not args.force and os.path.exists(bin_out) and os.path.exists(sub_out):
        print(f"[done] {save_dir} - explain_*.png already present")
        return

    print(f"\n=== {save_dir} ===")
    branches = branches_for(ablation) if ablation is not None else None
    model_kwargs = {'backbone': backbone, 'hub_version': hub_version}
    if branches is not None or ablation is not None:
        model_kwargs['attention_branches'] = branches
    model = MultiTaskOralClassifier(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if args.force or not os.path.exists(bin_out):
        print("[1/2] Binary head...")
        bin_samples = collect_diverse_samples(test_loader, args.num_samples,
                                                num_classes=2, label_idx=1)
        build_panel(model, backbone, bin_samples, device, head='binary',
                    save_path=bin_out, lime_samples=args.lime_samples,
                    skip_gradcam=args.skip_gradcam, skip_lime=args.skip_lime)

    if args.force or not os.path.exists(sub_out):
        print("[2/2] Subtype head...")
        sub_samples = collect_diverse_samples(test_loader, args.num_samples,
                                                num_classes=len(DS2_CLASSES),
                                                label_idx=2)
        build_panel(model, backbone, sub_samples, device, head='subtype',
                    save_path=sub_out, lime_samples=args.lime_samples,
                    skip_gradcam=args.skip_gradcam, skip_lime=args.skip_lime)


def main():
    parser = argparse.ArgumentParser(description='LIME + GradCAM explainability')
    parser.add_argument('--backbone', type=str, default=BACKBONE)
    parser.add_argument('--ablation', type=str, default=None,
                        choices=sorted(ABLATIONS.keys()),
                        help='Ablation key (custom_efficientnet_v2 only)')
    parser.add_argument('--recipe', type=str, default='tuned',
                        choices=['tuned', 'baseline'])
    parser.add_argument('--hub-version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='AttentionHub variant. v2 = sequential Triplet->EMA.')
    parser.add_argument('--all-missing', action='store_true',
                        help='Loop every results/* folder; skip those with explain_*.png')
    parser.add_argument('--force', action='store_true',
                        help='Re-generate even if explain_*.png exists')
    parser.add_argument('--num-samples', type=int, default=2,
                        help='Samples per class to explain')
    parser.add_argument('--lime-samples', type=int, default=500,
                        help='Perturbations per LIME explanation (lower = faster, noisier)')
    parser.add_argument('--skip-lime', action='store_true')
    parser.add_argument('--skip-gradcam', action='store_true')
    args = parser.parse_args()

    if not args.skip_gradcam and not HAS_GRADCAM:
        print("pytorch-grad-cam not installed. Run: pip install grad-cam")
        sys.exit(1)
    if not args.skip_lime and not HAS_LIME:
        print("lime not installed. Run: pip install lime scikit-image")
        sys.exit(1)

    set_seed()
    device = get_device()

    print("Loading test set...")
    d1p, d1b, d1s = load_dataset1_split('test')
    d2p, d2b, d2s = load_dataset2_split('test')
    test_ds = OralPathologyDataset(d1p + d2p, d1b + d2b, d1s + d2s,
                                    transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=min(BATCH_SIZE, 32),
                              shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test images: {len(test_ds)}")

    if args.all_missing:
        results_root = os.path.join(config.BASE_PATH, 'results')
        for name in sorted(os.listdir(results_root)):
            full = os.path.join(results_root, name)
            if not os.path.isdir(full):
                continue
            parsed = _parse_run_folder(name)
            if parsed is None:
                print(f"[skip] {name} - unrecognised folder pattern")
                continue
            bb, ab, rc, hv = parsed
            _explain_run(bb, ab, rc, device, test_loader, args, hub_version=hv)
    else:
        _explain_run(args.backbone, args.ablation, args.recipe,
                     device, test_loader, args, hub_version=args.hub_version)

    print("\nDone.")


if __name__ == "__main__":
    main()
