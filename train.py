"""
Training script.

For custom_efficientnet_v2 this script automatically applies the better
training recipe from custom_efficientnet_colab.py:
  - Kaiming initialization on AttentionHub + heads
  - AMP (mixed precision) with GradScaler
  - Gradient clipping (norm=1.0)
  - Linear LR warmup (3 epochs: 1e-5 -> 1e-3)
  - CosineAnnealingLR after warmup
  - Early-stopping patience = 25

All other models use the standard training recipe from configs/config.py.

NOTE: Test set is NOT touched here.
      Run evaluate_final.py once after training.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import (
    NUM_WORKERS, BATCH_SIZE, NUM_SUBTYPES, BACKBONE, DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, SAVE_DIR, BEST_MODEL_PATH,
    SCHEDULER_TYPE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
    EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
)
from configs import config
from utils.common import set_seed, get_device
from data.transforms import train_transform, val_transform
from data.dataset import OralPathologyDataset, load_dataset1_split, load_dataset2_split
from models.architecture import MultiTaskOralClassifier
from models.loss import MultiTaskLoss
from engine.trainer import train_one_epoch, validate
from utils.ablation import ABLATIONS, branches_for, run_name_for

# ?? Custom EfficientNetV2 training hyper-parameters (from colab) ??????????
_CUSTOM_LR            = 1e-3
_CUSTOM_WARMUP_EPOCHS = 3
_CUSTOM_GRAD_CLIP     = 1.0
_CUSTOM_ES_PATIENCE   = 25

# Gradient-accumulation overrides — heavy backbones OOM at BATCH_SIZE=64.
# To preserve fair comparison the DataLoader batch size stays at the global
# BATCH_SIZE (so shuffle order, total steps, and effective gradient are
# identical), but the forward/backward pass for these models is split into
# micro-batches of (BATCH_SIZE / accum_steps) and accumulated before
# optimizer.step(). Mathematically equivalent to a real batch=BATCH_SIZE
# pass — only peak activation memory is reduced.
#
# VGG19 (~143M params) reliably OOM-ed at batch=64 on consumer GPUs, leaving
# results/vgg19/ empty after run_all_models.py.
_GRAD_ACCUM_OVERRIDES = {
    'vgg19': 4,   # micro-batch = 64/4 = 16
}


def _apply_kaiming_init(module):
    """Kaiming normal init for Conv2d, BN, and Linear layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def main():
    set_seed()
    device = get_device()

    parser = argparse.ArgumentParser(description='Train Oral Pathology Model')
    parser.add_argument('--backbone', type=str, default=BACKBONE,
                        help='Backbone model name')
    parser.add_argument('--recipe', type=str, default='tuned',
                        choices=['tuned', 'baseline'],
                        help='Training recipe. "tuned" = per-model best (custom uses '
                             'warmup/AMP/clip/Kaiming/ES=25). "baseline" = forces the '
                             'standard 8-model recipe even for the custom model '
                             '(ablation for fair comparison).')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=sorted(ABLATIONS.keys()),
                        help='AttentionHub ablation (custom_efficientnet_v2 only). '
                             'Forces --recipe baseline for fairness. "none" replaces '
                             'the hub with the donor EfficientNetV2-B0 Block-4 (no-attention '
                             'control). "full" is the proposed model.')
    parser.add_argument('--hub-version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='AttentionHub variant (custom_efficientnet_v2 only). '
                             'v1 = parallel BAM/Triplet/KAN concat-fuse (default, '
                             'compatible with --ablation). v2 = sequential '
                             'Triplet->EMA cascade with LayerScale gates (informed '
                             'by ablation; forces --recipe baseline).')
    args = parser.parse_args()
    current_backbone = args.backbone
    recipe = args.recipe
    ablation = args.ablation
    hub_version = args.hub_version
    if hub_version == 'v2':
        if ablation is not None:
            raise SystemExit("--hub-version v2 is incompatible with --ablation")
        if recipe != 'baseline':
            print(f"NOTE: --hub-version v2 forces --recipe baseline (was {recipe!r}).")
        recipe = 'baseline'
    if ablation is not None:
        # Ablations are only meaningful under the fair (baseline) recipe.
        if recipe != 'baseline':
            print(f"NOTE: --ablation forces --recipe baseline (was {recipe!r}).")
        recipe = 'baseline'

    accum_steps = _GRAD_ACCUM_OVERRIDES.get(current_backbone, 1)
    print(f"Using Backbone: {current_backbone} | Recipe: {recipe} | "
          f"Batch: {BATCH_SIZE} | Grad-accum: {accum_steps} "
          f"(effective batch = {BATCH_SIZE})")

    # "is_custom" governs whether the custom training recipe is applied.
    # Under --recipe baseline we deliberately disable it even for the custom model
    # so the ablation row uses the exact same recipe as the 8 baselines.
    is_custom_arch = (current_backbone == 'custom_efficientnet_v2')
    is_custom      = is_custom_arch and (recipe == 'tuned')

    # Resolve the per-run results folder name (handles recipe + ablation + hub).
    run_name = run_name_for(current_backbone, ablation, recipe, hub_version=hub_version)

    # Paths
    current_save_dir       = os.path.join(config.BASE_PATH, 'results', run_name)
    current_best_model_path = os.path.join(current_save_dir, 'best_model.pth')
    os.makedirs(current_save_dir, exist_ok=True)
    print(f"Results will be saved to: {current_save_dir}")

    # ?? Data ?????????????????????????????????????????????????????????????
    print("Loading datasets (train + val only - test set held out)...")

    d1_train_p, d1_train_b, d1_train_s = load_dataset1_split('train')
    d1_val_p,   d1_val_b,   d1_val_s   = load_dataset1_split('val')
    d2_train_p, d2_train_b, d2_train_s = load_dataset2_split('train')
    d2_val_p,   d2_val_b,   d2_val_s   = load_dataset2_split('val')

    train_ds = OralPathologyDataset(
        d1_train_p + d2_train_p, d1_train_b + d2_train_b, d1_train_s + d2_train_s,
        transform=train_transform)
    val_ds = OralPathologyDataset(
        d1_val_p + d2_val_p, d1_val_b + d2_val_b, d1_val_s + d2_val_s,
        transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ?? Model ?????????????????????????????????????????????????????????????
    model = MultiTaskOralClassifier(
        backbone=current_backbone,
        attention_branches=branches_for(ablation) if ablation is not None else None,
        hub_version=hub_version,
    ).to(device)

    if is_custom:
        # Kaiming init for the custom attention hub + classification heads
        model.backbone.stage4.apply(_apply_kaiming_init)
        model.head_binary.apply(_apply_kaiming_init)
        model.head_subtype.apply(_apply_kaiming_init)
        print("Kaiming init applied to AttentionHub + classification heads")

    criterion = MultiTaskLoss()
    lr_init   = _CUSTOM_LR if is_custom else LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=WEIGHT_DECAY)

    # AMP scaler (custom model only - trains from scratch without pretrained weights)
    scaler = None
    if is_custom and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("AMP (mixed precision) enabled")

    grad_clip = _CUSTOM_GRAD_CLIP if is_custom else None

    # Scheduler
    es_patience = _CUSTOM_ES_PATIENCE if is_custom else EARLY_STOPPING_PATIENCE

    if is_custom:
        # CosineAnnealingLR kicks in after warmup
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - _CUSTOM_WARMUP_EPOCHS, eta_min=1e-6)
        scheduler = None          # managed manually below
        print(f"Scheduler: warmup({_CUSTOM_WARMUP_EPOCHS} ep) + CosineAnnealing")
    else:
        cosine_scheduler = None
        if SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        elif SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
        elif SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_FACTOR)
        elif SCHEDULER_TYPE == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
        else:
            scheduler = None
        print(f"Scheduler: {SCHEDULER_TYPE}")

    if EARLY_STOPPING:
        print(f"Early stopping: patience={es_patience}, min_delta={EARLY_STOPPING_MIN_DELTA}")

    # ?? Training loop ?????????????????????????????????????????????????????
    best_loss      = float('inf')
    epochs_no_impr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc_b': [], 'val_acc_s': [], 'lr': [],
               'epoch_time_s': []}

    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # Linear LR warmup for custom model
        if is_custom and epoch < _CUSTOM_WARMUP_EPOCHS:
            warmup_lr = 1e-5 + (_CUSTOM_LR - 1e-5) * (epoch / _CUSTOM_WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']
        warmup_tag = " [warmup]" if (is_custom and epoch < _CUSTOM_WARMUP_EPOCHS) else ""
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.2e}{warmup_tag}")

        t_loss, t_loss_b, t_loss_s, t_acc_b, t_acc_s = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip_norm=grad_clip, accum_steps=accum_steps)
        v_loss, v_acc_b, v_acc_s = validate(model, val_loader, criterion, device)

        # Step schedulers
        if is_custom and epoch >= _CUSTOM_WARMUP_EPOCHS:
            cosine_scheduler.step()
        elif scheduler is not None:
            if SCHEDULER_TYPE == 'plateau':
                scheduler.step(v_loss)
            else:
                scheduler.step()

        epoch_elapsed = time.time() - epoch_start_time

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc_b'].append(v_acc_b)
        history['val_acc_s'].append(v_acc_s)
        history['lr'].append(current_lr)
        history['epoch_time_s'].append(round(epoch_elapsed, 2))

        print(f"Train Loss: {t_loss:.4f} (B:{t_loss_b:.4f} S:{t_loss_s:.4f}) | "
              f"Acc B:{t_acc_b:.4f} S:{t_acc_s:.4f}")
        print(f"Val   Loss: {v_loss:.4f} | Acc B:{v_acc_b:.4f} S:{v_acc_s:.4f}")

        if v_loss < (best_loss - EARLY_STOPPING_MIN_DELTA):
            best_loss      = v_loss
            epochs_no_impr = 0
            torch.save(model.state_dict(), current_best_model_path)
            print("OK Best model saved")
        else:
            epochs_no_impr += 1
            if epochs_no_impr > 0:
                print(f"No improvement for {epochs_no_impr} epoch(s)")

        # Early stopping (for custom model: only after warmup)
        start_es = _CUSTOM_WARMUP_EPOCHS if is_custom else 0
        if EARLY_STOPPING and epoch >= start_es and epochs_no_impr >= es_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    total_training_time = time.time() - training_start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Epochs: {epoch+1}/{NUM_EPOCHS} | Best val loss: {best_loss:.4f}")
    print(f"Total training time: {total_training_time:.1f}s "
          f"({total_training_time/60:.1f} min)")
    print(f"Model saved: {current_best_model_path}")
    print("\nTo evaluate on the test set run:")
    eval_cmd = f"  python evaluate_final.py --backbone {current_backbone}"
    if is_custom_arch and recipe == 'baseline':
        eval_cmd += " --recipe baseline"
    if hub_version == 'v2':
        eval_cmd += " --hub-version v2"
    if ablation is not None:
        eval_cmd += f" --ablation {ablation}"
    print(eval_cmd)
    print("=" * 60)

    # ?? Save training time metrics ????????????????????????????????????????
    training_metrics = {
        'backbone': current_backbone,
        'recipe': recipe,
        'hub_version': hub_version,
        'ablation': ablation,
        'total_training_time_s': round(total_training_time, 2),
        'total_training_time_min': round(total_training_time / 60, 2),
        'epochs_completed': epoch + 1,
        'epochs_max': NUM_EPOCHS,
        'best_val_loss': round(best_loss, 4),
        'epoch_times_s': history['epoch_time_s'],
        'avg_epoch_time_s': round(sum(history['epoch_time_s']) / len(history['epoch_time_s']), 2),
    }
    timing_file = os.path.join(current_save_dir, 'training_time.json')
    with open(timing_file, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"OK Training time metrics saved: {timing_file}")


if __name__ == "__main__":
    main()
