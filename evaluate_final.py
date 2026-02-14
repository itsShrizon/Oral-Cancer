"""
FINAL TEST SET EVALUATION
⚠️  Run this ONLY ONCE after training is complete!
Multiple runs on the test set lead to overfitting through hyperparameter tuning.
"""
import torch
from torch.utils.data import DataLoader
from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT, 
    SAVE_DIR, BEST_MODEL_PATH, DS2_CLASSES
)
from utils.common import set_seed, get_device
from data.transforms import val_transform
from data.dataset import (
    OralPathologyDataset, load_dataset1_split, load_dataset2_split
)
from models.architecture import MultiTaskOralClassifier
from utils.evaluation import evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

def main():
    print("\n" + "="*70)
    print("⚠️  FINAL TEST SET EVALUATION - USE ONLY ONCE!")
    print("="*70)
    print("This script evaluates on the held-out test set.")
    print("Running this multiple times compromises the validity of results.\n")
    
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Oral Pathology Model')
    parser.add_argument('--backbone', type=str, default=BACKBONE, help='Backbone model name')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    current_backbone = args.backbone
    print(f"Using Backbone: {current_backbone}")

    if not args.no_confirm:
        response = input("Continue with test evaluation? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Evaluation cancelled.")
            return
    else:
        print("Skipping confirmation prompt (--no-confirm used).")
    
    # Update paths
    import os
    from configs import config
    
    # Recalculate paths based on the chosen backbone
    current_save_dir = os.path.join(config.BASE_PATH, 'results', current_backbone)
    current_best_model_path = os.path.join(current_save_dir, 'best_model.pth')
    
    if not os.path.exists(current_best_model_path):
        print(f"❌ Model file not found at {current_best_model_path}")
        print(f"   Make sure you have trained the {current_backbone} model first.")
        return

    set_seed()
    device = get_device()
    
    # Load test data ONLY
    print("\nLoading held-out test datasets...")
    d1_test_p, d1_test_b, d1_test_s = load_dataset1_split('test')
    d2_test_p, d2_test_b, d2_test_s = load_dataset2_split('test')  # Now from merged+split data
    
    test_ds_combined = OralPathologyDataset(
        d1_test_p + d2_test_p, 
        d1_test_b + d2_test_b, 
        d1_test_s + d2_test_s, 
        transform=val_transform
    )
    
    test_loader = DataLoader(test_ds_combined, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Test images: {len(test_ds_combined)}")
    print(f"  - Dataset 1: {len(d1_test_p)}")
    print(f"  - Dataset 2: {len(d2_test_p)}")
    
    # Load model
    print(f"\nLoading model from {current_best_model_path}...")
    model = MultiTaskOralClassifier(backbone=current_backbone).to(device)
    model.load_state_dict(torch.load(current_best_model_path))
    
    # Evaluate
    print("\nEvaluating on held-out test set...")
    results = evaluate_model(model, test_loader, device)
    
    # Print Results
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION RESULTS (Benign vs Malignant)")
    print("="*60)
    acc_b = accuracy_score(results['targets_binary'], results['preds_binary'])
    prec_b = precision_score(results['targets_binary'], results['preds_binary'], average='weighted', zero_division=0)
    rec_b = recall_score(results['targets_binary'], results['preds_binary'], average='weighted', zero_division=0)
    f1_b = f1_score(results['targets_binary'], results['preds_binary'], average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc_b:.4f}")
    print(f"Precision: {prec_b:.4f}")
    print(f"Recall:    {rec_b:.4f}")
    print(f"F1-Score:  {f1_b:.4f}")
    
    print("\n" + "="*60)
    print("SUBTYPE CLASSIFICATION RESULTS")
    print("="*60)
    acc_s = accuracy_score(results['targets_subtype'], results['preds_subtype'])
    prec_s = precision_score(results['targets_subtype'], results['preds_subtype'], average='weighted', zero_division=0)
    rec_s = recall_score(results['targets_subtype'], results['preds_subtype'], average='weighted', zero_division=0)
    f1_s = f1_score(results['targets_subtype'], results['preds_subtype'], average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc_s:.4f}")
    print(f"Precision: {prec_s:.4f}")
    print(f"Recall:    {rec_s:.4f}")
    print(f"F1-Score:  {f1_s:.4f}")
    
    print("\nPer-Class Report:")
    print(classification_report(results['targets_subtype'], results['preds_subtype'], 
                                 target_names=DS2_CLASSES, zero_division=0))
    
    # Save results to file
    results_file = os.path.join(current_save_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("BINARY CLASSIFICATION RESULTS (Benign vs Malignant)\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {acc_b:.4f}\n")
        f.write(f"Precision: {prec_b:.4f}\n")
        f.write(f"Recall:    {rec_b:.4f}\n")
        f.write(f"F1-Score:  {f1_b:.4f}\n\n")
        
        f.write("SUBTYPE CLASSIFICATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:  {acc_s:.4f}\n")
        f.write(f"Precision: {prec_s:.4f}\n")
        f.write(f"Recall:    {rec_s:.4f}\n")
        f.write(f"F1-Score:  {f1_s:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write(classification_report(results['targets_subtype'], results['preds_subtype'], 
                                      target_names=DS2_CLASSES, zero_division=0))
    
    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
