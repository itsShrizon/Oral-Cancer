"""
Generate detailed prediction visualizations and confusion matrices
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import random

from configs.config import (
    NUM_SUBTYPES, BACKBONE, DROPOUT, BEST_MODEL_PATH, 
    DS2_CLASSES, SAVE_DIR, BATCH_SIZE, NUM_WORKERS
)
from utils.common import set_seed, get_device
from data.transforms import val_transform
from data.dataset import (
    OralPathologyDataset, load_dataset1_split, load_dataset2_split
)
from models.architecture import MultiTaskOralClassifier
from torch.utils.data import DataLoader


def visualize_predictions(model, test_loader, device, num_samples=16, save_dir=None):
    """
    Visualize model predictions on sample images.
    """
    model.eval()
    
    # Get a batch
    images, targets_binary, targets_subtype = next(iter(test_loader))
    
    # Select random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    images = images[indices]
    targets_binary = targets_binary[indices]
    targets_subtype = targets_subtype[indices]
    
    images_device = images.to(device)
    
    with torch.no_grad():
        pred_binary, pred_subtype = model(images_device)
        preds_b = torch.argmax(pred_binary, dim=1).cpu()
        preds_s = torch.argmax(pred_subtype, dim=1).cpu()
    
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images * std + mean
    
    binary_labels = ['Benign', 'Malignant']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 18))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images_denorm[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        true_b = binary_labels[targets_binary[i]]
        pred_b = binary_labels[preds_b[i]]
        
        true_s = DS2_CLASSES[targets_subtype[i]] if targets_subtype[i] != -1 else 'N/A'
        pred_s = DS2_CLASSES[preds_s[i]] if targets_subtype[i] != -1 else 'N/A'
        
        color = 'green' if preds_b[i] == targets_binary[i] else 'red'
        
        axes[i].set_title(
            f"True: {true_b} / {true_s}\nPred: {pred_b} / {pred_s}",
            color=color, fontsize=10
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir or SAVE_DIR, 'prediction_samples.png')
    plt.savefig(save_path, dpi=150)
    print(f"✓ Prediction samples saved to {save_path}")
    plt.show()


def plot_confusion_matrices(model, test_loader, device, save_dir=None):
    """
    Generate confusion matrices for both binary and subtype classification.
    """
    model.eval()
    
    all_targets_b, all_preds_b = [], []
    all_targets_s, all_preds_s = [], []
    
    with torch.no_grad():
        for images, targets_b, targets_s in test_loader:
            images = images.to(device)
            pred_b, pred_s = model(images)
            
            preds_b = torch.argmax(pred_b, dim=1)
            all_preds_b.extend(preds_b.cpu().numpy())
            all_targets_b.extend(targets_b.numpy())
            
            mask = targets_s != -1
            if mask.sum() > 0:
                preds_s = torch.argmax(pred_s[mask], dim=1)
                all_preds_s.extend(preds_s.cpu().numpy())
                all_targets_s.extend(targets_s[mask].numpy())
    
    # Convert to numpy arrays
    all_targets_b = np.array(all_targets_b)
    all_preds_b = np.array(all_preds_b)
    all_targets_s = np.array(all_targets_s)
    all_preds_s = np.array(all_preds_s)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Binary confusion matrix
    cm_binary = confusion_matrix(all_targets_b, all_preds_b)
    binary_labels = ['Benign', 'Malignant']
    
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                xticklabels=binary_labels, yticklabels=binary_labels,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Binary Classification Confusion Matrix\n(Benign vs Malignant)', 
                  fontsize=14, fontweight='bold')
    
    # Calculate accuracies for binary
    binary_acc = (cm_binary[0, 0] + cm_binary[1, 1]) / cm_binary.sum()
    ax1.text(0.5, -0.15, f'Overall Accuracy: {binary_acc:.2%}', 
             ha='center', va='top', transform=ax1.transAxes, fontsize=11)
    
    # Subtype confusion matrix
    cm_subtype = confusion_matrix(all_targets_s, all_preds_s)
    
    sns.heatmap(cm_subtype, annot=True, fmt='d', cmap='Greens',
                xticklabels=DS2_CLASSES, yticklabels=DS2_CLASSES,
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Subtype Classification Confusion Matrix\n(7 Classes)', 
                  fontsize=14, fontweight='bold')
    
    # Calculate accuracy for subtype
    subtype_acc = np.trace(cm_subtype) / cm_subtype.sum()
    ax2.text(0.5, -0.15, f'Overall Accuracy: {subtype_acc:.2%}', 
             ha='center', va='top', transform=ax2.transAxes, fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir or SAVE_DIR, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {save_path}")
    plt.show()


def generate_detailed_grid(model, test_dataset, device, num_samples=20, save_dir=None):
    """
    Generate a detailed grid showing random test samples with predictions.
    Similar to the notebook visualization.
    """
    model.eval()
    
    # Get random indices
    total_samples = len(test_dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    binary_labels = ['Benign', 'Malignant']
    
    for idx, sample_idx in enumerate(indices):
        img, target_b, target_s = test_dataset[sample_idx]
        
        # Make prediction
        img_device = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_binary, pred_subtype = model(img_device)
            pred_b = torch.argmax(pred_binary, dim=1).item()
            pred_s = torch.argmax(pred_subtype, dim=1).item()
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = img_denorm.permute(1, 2, 0).numpy()
        img_denorm = np.clip(img_denorm, 0, 1)
        
        # Display image
        axes[idx].imshow(img_denorm)
        
        # Get labels
        true_b = binary_labels[target_b]
        pred_b_label = binary_labels[pred_b]
        
        if target_s != -1:
            true_s = DS2_CLASSES[target_s]
            pred_s_label = DS2_CLASSES[pred_s]
        else:
            true_s = 'N/A'
            pred_s_label = 'N/A'
        
        # Determine color (green if correct, red if wrong)
        correct_binary = (pred_b == target_b)
        correct_subtype = (pred_s == target_s) if target_s != -1 else True
        color = 'green' if (correct_binary and correct_subtype) else 'red'
        
        # Create title
        title = f"True: {true_b} / {true_s}\nPred: {pred_b_label} / {pred_s_label}"
        axes[idx].set_title(title, color=color, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide remaining subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir or SAVE_DIR, 'detailed_prediction_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Detailed prediction grid saved to {save_path}")
    plt.show()


def main():
    import argparse
    from configs import config

    parser = argparse.ArgumentParser(description='Generate prediction visualizations')
    parser.add_argument('--backbone', type=str, default=BACKBONE, help='Backbone model name')
    args = parser.parse_args()

    current_backbone = args.backbone
    current_save_dir = os.path.join(config.BASE_PATH, 'results', current_backbone)
    current_best_model_path = os.path.join(current_save_dir, 'best_model.pth')

    print("="*70)
    print(f"GENERATING PREDICTION VISUALIZATIONS & CONFUSION MATRICES ({current_backbone})")
    print("="*70)

    if not os.path.exists(current_best_model_path):
        print(f"❌ Model file not found at {current_best_model_path}")
        print(f"   Make sure you have trained the {current_backbone} model first.")
        return

    set_seed()
    device = get_device()

    # Load test data
    print("\nLoading test datasets...")
    d1_test_p, d1_test_b, d1_test_s = load_dataset1_split('test')
    d2_test_p, d2_test_b, d2_test_s = load_dataset2_split('test')

    test_ds = OralPathologyDataset(
        d1_test_p + d2_test_p,
        d1_test_b + d2_test_b,
        d1_test_s + d2_test_s,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Test images: {len(test_ds)}")

    # Load model
    print(f"\nLoading model from {current_best_model_path}...")
    model = MultiTaskOralClassifier(backbone=current_backbone).to(device)
    model.load_state_dict(torch.load(current_best_model_path))
    print("Model loaded successfully!")

    # Generate visualizations
    print("\n1. Generating confusion matrices...")
    plot_confusion_matrices(model, test_loader, device, save_dir=current_save_dir)

    print("\n2. Generating prediction samples (16 images)...")
    visualize_predictions(model, test_loader, device, num_samples=16, save_dir=current_save_dir)

    print("\n3. Generating detailed prediction grid (20 images)...")
    generate_detailed_grid(model, test_ds, device, num_samples=20, save_dir=current_save_dir)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nSaved to {current_save_dir}/:")
    print("  - confusion_matrices.png")
    print("  - prediction_samples.png")
    print("  - detailed_prediction_grid.png")


if __name__ == "__main__":
    main()
