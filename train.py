"""
Training script - Does NOT evaluate on test set.
Test set should only be evaluated once at the very end using evaluate_final.py
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT, 
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, SAVE_DIR, BEST_MODEL_PATH,
    SCHEDULER_TYPE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
    EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)
from utils.common import set_seed, get_device
from data.transforms import train_transform, val_transform
from data.dataset import (
    OralPathologyDataset, load_dataset1_split, load_dataset2_split
)
from models.architecture import MultiTaskOralClassifier
from models.loss import MultiTaskLoss
from engine.trainer import train_one_epoch, validate

def main():
    set_seed()
    device = get_device()
    
    # Argument Parsing
    import argparse
    parser = argparse.ArgumentParser(description='Train Oral Pathology Model')
    parser.add_argument('--backbone', type=str, default=BACKBONE, help='Backbone model name')
    args = parser.parse_args()
    
    current_backbone = args.backbone
    print(f"Using Backbone: {current_backbone}")
    
    # Update paths if backbone changed from config default
    import os
    from configs import config
    
    # Recalculate paths based on the chosen backbone
    # Note: We use config.BASE_PATH to ensure we are relative to the workspace root
    current_save_dir = os.path.join(config.BASE_PATH, 'results', current_backbone)
    current_best_model_path = os.path.join(current_save_dir, 'best_model.pth')
    
    os.makedirs(current_save_dir, exist_ok=True)
    print(f"Results will be saved to: {current_save_dir}")
    
    # 1. Load Datasets - ONLY TRAIN AND VAL (NO TEST!)
    print("Loading datasets...")
    print("⚠️  Test set is NOT loaded during training to prevent data leakage!")
    
    # Dataset 1 (Binary only)
    d1_train_p, d1_train_b, d1_train_s = load_dataset1_split('train')
    d1_val_p, d1_val_b, d1_val_s = load_dataset1_split('val')
    
    # Dataset 2 (Both labels) - now properly split from merged Training+Validation
    d2_train_p, d2_train_b, d2_train_s = load_dataset2_split('train')
    d2_val_p, d2_val_b, d2_val_s = load_dataset2_split('val')
    
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
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")
    print(f"Test images: Not loaded (use evaluate_final.py after training)")
    
    # 2. Model Setup
    model = MultiTaskOralClassifier(backbone=current_backbone).to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Setup scheduler based on configuration
    if SCHEDULER_TYPE == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        print(f"Scheduler: CosineAnnealingLR")
    elif SCHEDULER_TYPE == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, 
                                                         patience=SCHEDULER_PATIENCE, verbose=True)
        print(f"Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")
    elif SCHEDULER_TYPE == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_FACTOR)
        print(f"Scheduler: StepLR (step_size={SCHEDULER_STEP_SIZE}, gamma={SCHEDULER_FACTOR})")
    elif SCHEDULER_TYPE == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
        print(f"Scheduler: ExponentialLR (gamma={SCHEDULER_GAMMA})")
    else:
        scheduler = None
        print("No scheduler used")
    
    # 3. Training Loop
    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc_b': [], 'val_acc_s': [], 'lr': []}
    
    if EARLY_STOPPING:
        print(f"Early stopping enabled (patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA})")
    
    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.2e}")
        
        t_loss, t_loss_b, t_loss_s, t_acc_b, t_acc_s = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc_b, v_acc_s = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        if scheduler is not None:
            if SCHEDULER_TYPE == 'plateau':
                scheduler.step(v_loss)
            else:
                scheduler.step()
        
        # Track history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc_b'].append(v_acc_b)
        history['val_acc_s'].append(v_acc_s)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {t_loss:.4f} (B: {t_loss_b:.4f}, S: {t_loss_s:.4f}) | Acc B: {t_acc_b:.4f}, S: {t_acc_s:.4f}")
        print(f"Val Loss: {v_loss:.4f} | Acc B: {v_acc_b:.4f}, S: {v_acc_s:.4f}")
        
        # Check for improvement
        if v_loss < (best_loss - EARLY_STOPPING_MIN_DELTA):
            best_loss = v_loss
            epochs_no_improve = 0
            best_loss = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), current_best_model_path)
            print("✓ Best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve > 0:
                print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping check
        if EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total epochs: {epoch+1}/{NUM_EPOCHS}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {current_best_model_path}")
    if EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Stopped early due to no improvement")
    print("\n⚠️  IMPORTANT: To evaluate on test set, run:")
    print("   python evaluate_final.py")
    print("\nThis ensures test set is only used ONCE for final evaluation.")

if __name__ == "__main__":
    main()
