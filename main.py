import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT, 
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, SAVE_DIR, BEST_MODEL_PATH
)
from utils.common import set_seed, get_device, check_paths_exist
from data.transforms import train_transform, val_transform
from data.dataset import (
    OralPathologyDataset, load_dataset1_split, load_dataset2_split
)
from models.architecture import MultiTaskOralClassifier
from models.loss import MultiTaskLoss
from engine.trainer import train_one_epoch, validate
from utils.evaluation import evaluate_model
import os

def main():
    set_seed()
    device = get_device()
    
    # Check paths
    # Note: Accessing paths from dataset logic indirectly or verify config paths here if needed
    
    # 1. Load Datasets
    print("Loading datasets...")
    # Dataset 1 (Binary only)
    d1_train_p, d1_train_b, d1_train_s = load_dataset1_split('train')
    d1_val_p, d1_val_b, d1_val_s = load_dataset1_split('val')
    d1_test_p, d1_test_b, d1_test_s = load_dataset1_split('test')
    
    # Dataset 2 (Both labels)
    d2_train_p, d2_train_b, d2_train_s = load_dataset2_split('Training')
    d2_val_p, d2_val_b, d2_val_s = load_dataset2_split('Validation')
    d2_test_p, d2_test_b, d2_test_s = load_dataset2_split('Testing')
    
    # Combine
    train_paths = d1_train_p + d2_train_p
    train_binary = d1_train_b + d2_train_b
    train_subtype = d1_train_s + d2_train_s
    
    val_paths = d1_val_p + d2_val_p
    val_binary = d1_val_b + d2_val_b
    val_subtype = d1_val_s + d2_val_s
    
    # Create Datasets
    train_ds = OralPathologyDataset(train_paths, train_binary, train_subtype, transform=train_transform)
    val_ds = OralPathologyDataset(val_paths, val_binary, val_subtype, transform=val_transform)
    test_ds_combined = OralPathologyDataset(d1_test_p + d2_test_p, d1_test_b + d2_test_b, d1_test_s + d2_test_s, transform=val_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds_combined, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")
    
    # 2. Model Setup
    model = MultiTaskOralClassifier(num_subtypes=NUM_SUBTYPES, backbone=BACKBONE, dropout=DROPOUT).to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # 3. Training Loop
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        t_loss, t_loss_b, t_loss_s, t_acc_b, t_acc_s = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc_b, v_acc_s = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {t_loss:.4f} (B: {t_loss_b:.4f}, S: {t_loss_s:.4f}) | Acc B: {t_acc_b:.4f}, S: {t_acc_s:.4f}")
        print(f"Val Loss: {v_loss:.4f} | Acc B: {v_acc_b:.4f}, S: {v_acc_s:.4f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("✓ Best model saved")

    # 4. Evaluation
    print("\nEvaluating Best Model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    results = evaluate_model(model, test_loader, device)
    
    # Print Results
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
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
    from configs.config import DS2_CLASSES
    print(classification_report(results['targets_subtype'], results['preds_subtype'], 
                                 target_names=DS2_CLASSES, zero_division=0))
    
    # Save results to file
    results_file = os.path.join(SAVE_DIR, 'evaluation_results.txt')
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
    print(f"✓ Best model saved at {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
