"""
FINAL TEST SET EVALUATION
WARNING  Run this ONLY ONCE after training is complete!
Multiple runs on the test set lead to overfitting through hyperparameter tuning.

For custom_efficientnet_v2 this script automatically uses Test-Time Augmentation
(TTA) with 5 augmented views per image, matching the colab training recipe.
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT,
    SAVE_DIR, BEST_MODEL_PATH, DS2_CLASSES, IMG_SIZE,
)
from configs import config
from utils.common import set_seed, get_device
from data.transforms import val_transform
from data.dataset import OralPathologyDataset, load_dataset1_split, load_dataset2_split
from models.architecture import MultiTaskOralClassifier
from utils.evaluation import evaluate_model, plot_confusion_matrix

# ?? TTA transforms (same 5-view set used in custom_efficientnet_colab.py) ?
_tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def evaluate_model_tta(model, test_image_paths, test_binary, test_subtype, device):
    """Test-Time Augmentation: average softmax over 5 augmented views per image."""
    model.eval()
    results = {
        'preds_binary': [], 'targets_binary': [],
        'preds_subtype': [], 'targets_subtype': [],
    }
    print(f"Running TTA with {len(_tta_transforms)} views per image...")

    with torch.no_grad():
        for i in tqdm(range(len(test_image_paths)), desc="TTA Evaluation"):
            image    = Image.open(test_image_paths[i]).convert('RGB')
            target_b = test_binary[i]
            target_s = test_subtype[i]

            prob_b = torch.zeros(2,           device=device)
            prob_s = torch.zeros(NUM_SUBTYPES, device=device)

            for tfm in _tta_transforms:
                inp = tfm(image).unsqueeze(0).to(device)
                pred_b, pred_s = model(inp)
                prob_b += torch.softmax(pred_b, dim=1)[0]
                prob_s += torch.softmax(pred_s, dim=1)[0]

            results['preds_binary'].append(torch.argmax(prob_b).cpu().item())
            results['targets_binary'].append(target_b)
            if target_s != -1:
                results['preds_subtype'].append(torch.argmax(prob_s).cpu().item())
                results['targets_subtype'].append(target_s)

    return {k: np.array(v) for k, v in results.items()}


def main():
    print("\n" + "=" * 70)
    print("WARNING  FINAL TEST SET EVALUATION - USE ONLY ONCE!")
    print("=" * 70)

    parser = argparse.ArgumentParser(description='Evaluate Oral Pathology Model')
    parser.add_argument('--backbone',   type=str, default=BACKBONE,
                        help='Backbone model name')
    parser.add_argument('--recipe',     type=str, default='tuned',
                        choices=['tuned', 'baseline'],
                        help='Which trained checkpoint to evaluate. Must match the '
                             'recipe used during training. "baseline" disables TTA '
                             'for the custom model so the ablation is truly fair.')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Skip the confirmation prompt')
    args = parser.parse_args()

    current_backbone = args.backbone
    recipe           = args.recipe
    is_custom_arch   = (current_backbone == 'custom_efficientnet_v2')
    # TTA is part of the tuned recipe only.
    is_custom        = is_custom_arch and (recipe == 'tuned')

    run_name = current_backbone
    if is_custom_arch and recipe == 'baseline':
        run_name = f"{current_backbone}_baseline_recipe"

    print(f"Backbone: {current_backbone} | Recipe: {recipe}")
    if is_custom:
        print("TTA enabled (5 views per image - matches training recipe)")
    elif is_custom_arch and recipe == 'baseline':
        print("TTA disabled (baseline-recipe ablation - matches 8-model protocol)")

    if not args.no_confirm:
        resp = input("Continue with test evaluation? (yes/no): ")
        if resp.lower() not in ('yes', 'y'):
            print("Evaluation cancelled.")
            return
    else:
        print("Skipping confirmation (--no-confirm).")

    current_save_dir        = os.path.join(config.BASE_PATH, 'results', run_name)
    current_best_model_path = os.path.join(current_save_dir, 'best_model.pth')

    if not os.path.exists(current_best_model_path):
        print(f"FAIL Model not found: {current_best_model_path}")
        print(f"   Train the {current_backbone} model first.")
        return

    set_seed()
    device = get_device()

    # Load test data
    print("\nLoading held-out test dataset...")
    d1p, d1b, d1s = load_dataset1_split('test')
    d2p, d2b, d2s = load_dataset2_split('test')

    test_paths   = d1p + d2p
    test_binary  = d1b + d2b
    test_subtype = d1s + d2s
    print(f"Test images: {len(test_paths)}  (DS1: {len(d1p)}, DS2: {len(d2p)})")

    # Load model
    print(f"\nLoading model from {current_best_model_path}...")
    model = MultiTaskOralClassifier(backbone=current_backbone).to(device)
    model.load_state_dict(torch.load(current_best_model_path, map_location=device))

    # Evaluate
    print("\nEvaluating...")
    if is_custom:
        results = evaluate_model_tta(model, test_paths, test_binary, test_subtype, device)
    else:
        test_ds = OralPathologyDataset(test_paths, test_binary, test_subtype,
                                       transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        results = evaluate_model(model, test_loader, device)

    # ?? Binary results ????????????????????????????????????????????????????
    acc_b  = accuracy_score(results['targets_binary'],  results['preds_binary'])
    prec_b = precision_score(results['targets_binary'], results['preds_binary'],
                             average='weighted', zero_division=0)
    rec_b  = recall_score(results['targets_binary'],    results['preds_binary'],
                          average='weighted', zero_division=0)
    f1_b   = f1_score(results['targets_binary'],        results['preds_binary'],
                      average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION RESULTS (Benign vs Malignant)")
    print("=" * 60)
    print(f"Accuracy:  {acc_b:.4f}")
    print(f"Precision: {prec_b:.4f}")
    print(f"Recall:    {rec_b:.4f}")
    print(f"F1-Score:  {f1_b:.4f}")

    # ?? Subtype results ???????????????????????????????????????????????????
    acc_s  = accuracy_score(results['targets_subtype'],  results['preds_subtype'])
    prec_s = precision_score(results['targets_subtype'], results['preds_subtype'],
                             average='weighted', zero_division=0)
    rec_s  = recall_score(results['targets_subtype'],    results['preds_subtype'],
                          average='weighted', zero_division=0)
    f1_s   = f1_score(results['targets_subtype'],        results['preds_subtype'],
                      average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("SUBTYPE CLASSIFICATION RESULTS (7 classes)")
    print("=" * 60)
    print(f"Accuracy:  {acc_s:.4f}")
    print(f"Precision: {prec_s:.4f}")
    print(f"Recall:    {rec_s:.4f}")
    print(f"F1-Score:  {f1_s:.4f}")
    print("\nPer-Class Report:")
    print(classification_report(results['targets_subtype'], results['preds_subtype'],
                                 target_names=DS2_CLASSES, zero_division=0))

    # ?? Save results (TXT) ?????????????????????????????????????????????????
    results_file = os.path.join(current_save_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("BINARY CLASSIFICATION RESULTS (Benign vs Malignant)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:  {acc_b:.4f}\n")
        f.write(f"Precision: {prec_b:.4f}\n")
        f.write(f"Recall:    {rec_b:.4f}\n")
        f.write(f"F1-Score:  {f1_b:.4f}\n\n")
        f.write("SUBTYPE CLASSIFICATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:  {acc_s:.4f}\n")
        f.write(f"Precision: {prec_s:.4f}\n")
        f.write(f"Recall:    {rec_s:.4f}\n")
        f.write(f"F1-Score:  {f1_s:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(classification_report(results['targets_subtype'], results['preds_subtype'],
                                       target_names=DS2_CLASSES, zero_division=0))

    print(f"\nOK Results saved: {results_file}")

    # ?? Save results (JSON) ???????????????????????????????????????????????
    metrics_json = {
        'backbone': current_backbone,
        'recipe': recipe,
        'binary': {
            'accuracy':  round(acc_b, 4),
            'precision': round(prec_b, 4),
            'recall':    round(rec_b, 4),
            'f1_score':  round(f1_b, 4),
        },
        'subtype': {
            'accuracy':  round(acc_s, 4),
            'precision': round(prec_s, 4),
            'recall':    round(rec_s, 4),
            'f1_score':  round(f1_s, 4),
        },
    }
    json_file = os.path.join(current_save_dir, 'classification_metrics.json')
    with open(json_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"OK Classification metrics (JSON) saved: {json_file}")

    # ?? Confusion matrices ????????????????????????????????????????????????
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_confusion_matrix(
        results['targets_binary'], results['preds_binary'],
        ['Benign', 'Malignant'], 'Binary Classification', axes[0])

    plot_confusion_matrix(
        results['targets_subtype'], results['preds_subtype'],
        DS2_CLASSES, 'Subtype Classification (7-class)', axes[1])

    fig.suptitle(f'Confusion Matrices — {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_file = os.path.join(current_save_dir, 'confusion_matrices.png')
    fig.savefig(cm_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"OK Confusion matrices saved: {cm_file}")


if __name__ == "__main__":
    main()
