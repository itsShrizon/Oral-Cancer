# -*- coding: utf-8 -*-
"""
Custom EfficientNetV2 - Oral Cancer Multi-Task Classification

Complete pipeline for training and evaluating the CustomEfficientNetV2 model
on the Oral Cancer dataset using a dual-head multi-task setup
(Binary + Subtype classification).

Features vs standard train.py:
  - LR warmup (3 epochs: 1e-5 -> 1e-3)
  - CosineAnnealing scheduler (after warmup)
  - Kaiming initialization for AttentionHub + classification heads
  - AMP (mixed precision) with GradScaler
  - Gradient clipping
  - Test-Time Augmentation (TTA) at evaluation

Usage:
    python custom_efficientnet_colab.py
    python custom_efficientnet_colab.py --backbone custom_efficientnet_v2
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Configuration
# ============================================================
# Project root = directory containing this script
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Dataset 1 Paths
DS1_ORIGINAL_BENIGN = os.path.join(BASE_PATH, 'Dataset 1', 'original_data', 'benign_lesions')
DS1_ORIGINAL_MALIGNANT = os.path.join(BASE_PATH, 'Dataset 1', 'original_data', 'malignant_lesions')

DS2_TRAINING = os.path.join(BASE_PATH, 'Dataset 2', 'Training')
DS2_VALIDATION = os.path.join(BASE_PATH, 'Dataset 2', 'Validation')
DS2_TESTING = os.path.join(BASE_PATH, 'Dataset 2', 'Testing')

# Dataset configuration
DS2_CLASSES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
MALIGNANT_SUBTYPES = ['MC', 'OC', 'CaS']
NUM_SUBTYPES = len(DS2_CLASSES)

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 2
BACKBONE = 'custom_efficientnet_v2'
DROPOUT = 0.3
USE_PRETRAINED = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training configuration
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
SEED = 42
USE_AMP = True
GRAD_CLIP_NORM = 1.0
MIXUP_ALPHA = 0.0         # Disabled (harmful on small datasets)
LABEL_SMOOTHING = 0.0     # Disabled (harmful on small datasets)

# LR Warmup
WARMUP_EPOCHS = 3

# TTA (Test-Time Augmentation)
TTA_ENABLED = True

# Early Stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4

# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to {seed}")
    print(f"cuDNN benchmark enabled for faster convolutions")


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device


def check_paths_exist(paths_list):
    all_exist = True
    print("\nChecking paths...")
    for path in paths_list:
        exists = os.path.exists(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {path}")
        if not exists:
            all_exist = False
    return all_exist


def count_images_in_folder(folder):
    if not os.path.exists(folder):
        return 0
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    count = 0
    for ext in extensions:
        count += len(glob(os.path.join(folder, ext)))
        count += len(glob(os.path.join(folder, ext.upper())))
    return count


# ============================================================
# Data Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

# ============================================================
# Dataset & Data Loading
# ============================================================

class OralPathologyDataset(Dataset):
    """Union Dataset for Dual-Head Multi-Task Learning."""

    def __init__(self, image_paths, labels_binary, labels_subtype, transform=None):
        self.image_paths = image_paths
        self.labels_binary = labels_binary
        self.labels_subtype = labels_subtype
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels_binary[idx], self.labels_subtype[idx]


def get_image_files(folder):
    if not os.path.exists(folder):
        return []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(folder, ext)))
        files.extend(glob(os.path.join(folder, ext.upper())))
    return files


def load_dataset1_split(split='train', test_size=0.10, val_size=0.10, random_state=42):
    """
    Fixed split with properly separated train/val/test sets.
    Test set is completely held out and should ONLY be used for final evaluation.
    """
    benign_paths = get_image_files(DS1_ORIGINAL_BENIGN)
    malignant_paths = get_image_files(DS1_ORIGINAL_MALIGNANT)
    all_paths = benign_paths + malignant_paths
    all_binary = [0] * len(benign_paths) + [1] * len(malignant_paths)
    all_subtype = [-1] * len(all_paths)

    temp_paths, test_paths, temp_bin, test_bin, temp_sub, test_sub = train_test_split(
        all_paths, all_binary, all_subtype,
        test_size=test_size, random_state=random_state, stratify=all_binary
    )
    val_size_adj = val_size / (1 - test_size)
    train_paths, val_paths, train_bin, val_bin, train_sub, val_sub = train_test_split(
        temp_paths, temp_bin, temp_sub,
        test_size=val_size_adj, random_state=random_state, stratify=temp_bin
    )

    if split == 'train': return train_paths, train_bin, train_sub
    elif split == 'val':  return val_paths,   val_bin,   val_sub
    else:                 return test_paths,  test_bin,  test_sub


def load_dataset2_split(split='train', test_size=0.20, val_size=0.20, random_state=42):
    """
    Load Dataset 2 with proper train/val/test split.
    MERGES Training + Validation folders, then splits properly to avoid
    the suspicious pre-made Testing folder with identical distributions.
    """
    image_paths, labels_binary, labels_subtype = [], [], []

    for base_path in [DS2_TRAINING, DS2_VALIDATION]:
        for idx, subtype in enumerate(DS2_CLASSES):
            imgs = get_image_files(os.path.join(base_path, subtype))
            image_paths.extend(imgs)
            labels_subtype.extend([idx] * len(imgs))
            labels_binary.extend([1 if subtype in MALIGNANT_SUBTYPES else 0] * len(imgs))

    temp_paths, test_paths, temp_bin, test_bin, temp_sub, test_sub = train_test_split(
        image_paths, labels_binary, labels_subtype,
        test_size=test_size, random_state=random_state, stratify=labels_subtype
    )
    val_size_adj = val_size / (1 - test_size)
    train_paths, val_paths, train_bin, val_bin, train_sub, val_sub = train_test_split(
        temp_paths, temp_bin, temp_sub,
        test_size=val_size_adj, random_state=random_state, stratify=temp_sub
    )

    if split == 'train': return train_paths, train_bin, train_sub
    elif split == 'val':  return val_paths,   val_bin,   val_sub
    else:                 return test_paths,  test_bin,  test_sub


# ============================================================
# Model Architecture
# ============================================================

# --- BAM (Bottleneck Attention Module) ---

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True), nn.Linear(mid, channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        att = self.mlp(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        return att.expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x):
        return self.conv(x).expand_as(x)


class BAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(channels, reduction)

    def forward(self, x):
        return x * torch.sigmoid(self.channel_gate(x) + self.spatial_gate(x))


# --- Triplet Attention ---

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat([x.max(dim=1, keepdim=True).values,
                          x.mean(dim=1, keepdim=True)], dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            ZPool(),
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class TripletAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_ch = AttentionGate()
        self.gate_cw = AttentionGate()
        self.gate_hw = AttentionGate()

    def forward(self, x):
        x_ch = self.gate_ch(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        x_cw = self.gate_cw(x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        x_hw = self.gate_hw(x)
        return (x_ch + x_cw + x_hw) / 3.0


# --- KAN Attention ---

class BSplineActivation(nn.Module):
    def __init__(self, channels, num_bases=5, grid_range=3.0):
        super().__init__()
        self.h = (2 * grid_range) / (num_bases - 1)
        self.register_buffer("grid", torch.linspace(-grid_range, grid_range, num_bases))
        self.coeffs = nn.Parameter(torch.randn(channels, num_bases) * 0.1)

    def forward(self, x):
        bases = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.grid) / self.h).pow(2))
        return (bases * self.coeffs).sum(dim=-1) + F.silu(x)


class KANAttention(nn.Module):
    def __init__(self, channels, num_bases=5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.spline = BSplineActivation(channels, num_bases)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = torch.sigmoid(self.spline(self.gap(x).view(b, c)))
        return x * s.view(b, c, 1, 1)


# --- Attention Hub (Stage 4) ---

class AttentionHub(nn.Module):
    """Three-branch attention fusion: BAM + Triplet + KAN."""

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        branch_ch = in_channels // 2

        def reducer():
            return nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU(inplace=True),
            )

        self.reduce_bam = reducer()
        self.reduce_tri = reducer()
        self.reduce_kan = reducer()
        self.bam     = BAM(branch_ch, reduction)
        self.triplet = TripletAttention()
        self.kan     = KANAttention(branch_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([
            self.bam(self.reduce_bam(x)),
            self.triplet(self.reduce_tri(x)),
            self.kan(self.reduce_kan(x)),
        ], dim=1))


# --- Custom EfficientNetV2 Backbone ---

class CustomEfficientNetV2(nn.Module):
    """
    Custom 5-stage EfficientNetV2 with multi-branch attention.

    Stage layout:
      1. Stem + Block-0 + Block-1  (Fused-MBConv)    -> 32 ch
      2. Block-2                   (Fused-MBConv)    -> 48 ch
      3. Block-3                   (MBConv)          -> 96 ch
      4. AttentionHub (BAM+Triplet+KAN)              -> 112 ch
      5. Block-5                   (MBConv+SE)       -> 192 ch
      6. GAP -> flatten
    """

    num_features = 192

    def __init__(self, num_classes=2, pretrained=False, dropout=0.2, verbose=False):
        super().__init__()
        self.verbose = verbose
        self._num_classes = num_classes

        donor = timm.create_model("tf_efficientnetv2_b0", pretrained=pretrained, num_classes=0)
        self.stem   = nn.Sequential(donor.conv_stem, donor.bn1)
        self.stage1 = nn.Sequential(donor.blocks[0], donor.blocks[1])
        self.stage2 = donor.blocks[2]
        self.stage3 = donor.blocks[3]
        self.stage4 = AttentionHub(in_channels=96, out_channels=112)
        self.stage5 = donor.blocks[5]
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        del donor

        if num_classes > 0:
            self.dropout    = nn.Dropout(p=dropout)
            self.classifier = nn.Linear(192, num_classes)
        else:
            self.dropout    = None
            self.classifier = None

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = self.flatten(x)
        if self.classifier is not None:
            x = self.dropout(x)
            x = self.classifier(x)
        return x


# --- Multi-Task Classifier (Dual Head) ---

class MultiTaskOralClassifier(nn.Module):
    """Shared CustomEfficientNetV2 backbone with binary + subtype heads."""

    def __init__(self, num_subtypes=NUM_SUBTYPES, dropout=DROPOUT, pretrained=USE_PRETRAINED):
        super().__init__()
        self.backbone = CustomEfficientNetV2(num_classes=0, pretrained=pretrained)
        num_features = self.backbone.num_features  # 192
        self.dropout_layer = nn.Dropout(p=dropout)

        def head(out):
            return nn.Sequential(
                nn.Linear(num_features, 512), nn.ReLU(),
                nn.Dropout(p=dropout), nn.Linear(512, out)
            )

        self.head_binary  = head(2)
        self.head_subtype = head(num_subtypes)
        print(f"Model initialized: custom_efficientnet_v2 (features={num_features})")

    def forward(self, x):
        f = self.dropout_layer(self.backbone(x))
        return self.head_binary(f), self.head_subtype(f)

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True


# --- Multi-Task Loss ---

class MultiTaskLoss(nn.Module):
    def __init__(self, weight_binary=1.0, weight_subtype=1.0, label_smoothing=0.0):
        super().__init__()
        self.wb = weight_binary
        self.ws = weight_subtype
        self.crit_b = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.crit_s = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

    def forward(self, pred_b, pred_s, tgt_b, tgt_s):
        loss_b = self.crit_b(pred_b, tgt_b)
        loss_s = self.crit_s(pred_s, tgt_s)
        if torch.isnan(loss_s):
            loss_s = torch.tensor(0.0, device=pred_b.device)
        return self.wb * loss_b + self.ws * loss_s, loss_b, loss_s


# ============================================================
# Training & Validation Functions
# ============================================================

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def mixup_data(images, alpha=0.2):
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx]


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scaler=None, mixup_alpha=0.0, grad_clip_norm=None):
    model.train()
    running_loss = running_loss_b = running_loss_s = 0.0
    all_preds_b, all_targets_b, all_preds_s, all_targets_s = [], [], [], []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, tgt_b, tgt_s in pbar:
        images = images.to(device, non_blocking=True)
        tgt_b  = tgt_b.to(device, non_blocking=True)
        tgt_s  = tgt_s.to(device, non_blocking=True)

        if mixup_alpha > 0:
            images = mixup_data(images, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred_b, pred_s = model(images)
                loss, loss_b, loss_s = criterion(pred_b, pred_s, tgt_b, tgt_s)
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_b, pred_s = model(images)
            loss, loss_b, loss_s = criterion(pred_b, pred_s, tgt_b, tgt_s)
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss   += loss.item()
        running_loss_b += loss_b.item()
        running_loss_s += loss_s.item() if not torch.isnan(loss_s) else 0

        all_preds_b.extend(torch.argmax(pred_b, dim=1).cpu().numpy())
        all_targets_b.extend(tgt_b.cpu().numpy())

        mask = tgt_s != -1
        if mask.sum() > 0:
            all_preds_s.extend(torch.argmax(pred_s[mask], dim=1).cpu().numpy())
            all_targets_s.extend(tgt_s[mask].cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(train_loader)
    acc_s = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    return (running_loss / n, running_loss_b / n, running_loss_s / n,
            accuracy_score(all_targets_b, all_preds_b), acc_s)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds_b, all_targets_b, all_preds_s, all_targets_s = [], [], [], []
    use_amp = USE_AMP and device.type == 'cuda'

    with torch.no_grad():
        for images, tgt_b, tgt_s in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            tgt_b  = tgt_b.to(device, non_blocking=True)
            tgt_s  = tgt_s.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_b, pred_s = model(images)
                loss, _, _ = criterion(pred_b, pred_s, tgt_b, tgt_s)
            running_loss += loss.item()

            all_preds_b.extend(torch.argmax(pred_b, dim=1).cpu().numpy())
            all_targets_b.extend(tgt_b.cpu().numpy())

            mask = tgt_s != -1
            if mask.sum() > 0:
                all_preds_s.extend(torch.argmax(pred_s[mask], dim=1).cpu().numpy())
                all_targets_s.extend(tgt_s[mask].cpu().numpy())

    acc_s = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    return running_loss / len(val_loader), accuracy_score(all_targets_b, all_preds_b), acc_s


def evaluate_model(model, test_loader, device):
    """Standard evaluation without TTA."""
    model.eval()
    results = {'preds_binary': [], 'targets_binary': [], 'preds_subtype': [], 'targets_subtype': []}
    use_amp = USE_AMP and device.type == 'cuda'

    with torch.no_grad():
        for images, tgt_b, tgt_s in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_b, pred_s = model(images)
            results['preds_binary'].extend(torch.argmax(pred_b, dim=1).cpu().numpy())
            results['targets_binary'].extend(tgt_b.numpy())
            mask = tgt_s != -1
            if mask.sum() > 0:
                results['preds_subtype'].extend(torch.argmax(pred_s[mask], dim=1).cpu().numpy())
                results['targets_subtype'].extend(tgt_s[mask].numpy())

    return {k: np.array(v) for k, v in results.items()}


def evaluate_model_tta(model, test_image_paths, test_binary, test_subtype, device, tta_tfms):
    """Test-Time Augmentation: average softmax across N augmented views."""
    model.eval()
    results = {'preds_binary': [], 'targets_binary': [], 'preds_subtype': [], 'targets_subtype': []}
    use_amp = USE_AMP and device.type == 'cuda'
    print(f"Running TTA with {len(tta_tfms)} views per image...")

    with torch.no_grad():
        for i in tqdm(range(len(test_image_paths)), desc="TTA Evaluation"):
            image    = Image.open(test_image_paths[i]).convert('RGB')
            target_b = test_binary[i]
            target_s = test_subtype[i]

            prob_b_sum = torch.zeros(2, device=device)
            prob_s_sum = torch.zeros(NUM_SUBTYPES, device=device)

            for tfm in tta_tfms:
                tensor = tfm(image).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_b, pred_s = model(tensor)
                prob_b_sum += torch.softmax(pred_b, dim=1)[0]
                prob_s_sum += torch.softmax(pred_s, dim=1)[0]

            results['preds_binary'].append(torch.argmax(prob_b_sum).cpu().item())
            results['targets_binary'].append(target_b)
            if target_s != -1:
                results['preds_subtype'].append(torch.argmax(prob_s_sum).cpu().item())
                results['targets_subtype'].append(target_s)

    return {k: np.array(v) for k, v in results.items()}


def plot_confusion_matrix(y_true, y_pred, classes, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Custom EfficientNetV2 for Oral Cancer Classification')
    parser.add_argument('--backbone', type=str, default='custom_efficientnet_v2',
                        help='Backbone name (should be custom_efficientnet_v2)')
    args = parser.parse_args()

    backbone_name = args.backbone

    # Paths
    save_dir            = os.path.join(BASE_PATH, 'results', backbone_name)
    best_model_path     = os.path.join(save_dir, 'best_model.pth')
    history_plot_path   = os.path.join(save_dir, 'training_history.png')
    confusion_mat_path  = os.path.join(save_dir, 'confusion_matrices.png')
    results_file        = os.path.join(save_dir, 'evaluation_results.txt')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Results will be saved to: {save_dir}")
    print(f"IMG_SIZE: {IMG_SIZE} | Batch: {BATCH_SIZE} | AMP: {USE_AMP}")
    print(f"Warmup: {WARMUP_EPOCHS} epochs | Grad clip: {GRAD_CLIP_NORM} | TTA: {TTA_ENABLED}")

    set_seed()
    device = get_device()

    # Verify paths exist
    check_paths_exist([DS1_ORIGINAL_BENIGN, DS1_ORIGINAL_MALIGNANT, DS2_TRAINING, DS2_VALIDATION])

    # -------------------------------------------------------
    # Data Loading (train + val only - test set held out)
    # -------------------------------------------------------
    print("\nLoading datasets...")
    print("Test set is NOT loaded during training (prevents data leakage).")

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

    use_persistent = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=use_persistent)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=use_persistent)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    # -------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------
    model = MultiTaskOralClassifier().to(device)

    # Kaiming initialization for custom modules (better for from-scratch training)
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    model.backbone.stage4.apply(init_weights)
    model.head_binary.apply(init_weights)
    model.head_subtype.apply(init_weights)
    print("Kaiming init applied to AttentionHub + heads")

    total, trainable = count_params(model)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    criterion     = MultiTaskLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_val = MultiTaskLoss(label_smoothing=0.0)
    optimizer     = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    if scaler:
        print("AMP (mixed precision) enabled")

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    print(f"Scheduler: linear warmup ({WARMUP_EPOCHS} ep) + CosineAnnealing")
    if EARLY_STOPPING:
        print(f"Early stopping: patience={EARLY_STOPPING_PATIENCE}")

    if device.type == 'cuda':
        print(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    best_loss       = float('inf')
    epochs_no_impr  = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc_b': [], 'val_acc_s': [], 'lr': []}

    for epoch in range(NUM_EPOCHS):
        # Linear LR warmup
        if epoch < WARMUP_EPOCHS:
            warmup_lr = 1e-5 + (LEARNING_RATE - 1e-5) * (epoch / WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']
        suffix = " [warmup]" if epoch < WARMUP_EPOCHS else ""
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.2e}{suffix}")

        t_loss, t_loss_b, t_loss_s, t_acc_b, t_acc_s = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, mixup_alpha=MIXUP_ALPHA, grad_clip_norm=GRAD_CLIP_NORM)
        v_loss, v_acc_b, v_acc_s = validate(model, val_loader, criterion_val, device)

        if epoch >= WARMUP_EPOCHS:
            cosine_scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc_b'].append(v_acc_b)
        history['val_acc_s'].append(v_acc_s)
        history['lr'].append(current_lr)

        print(f"Train Loss: {t_loss:.4f} (B:{t_loss_b:.4f} S:{t_loss_s:.4f}) | "
              f"Acc B:{t_acc_b:.4f} S:{t_acc_s:.4f}")
        print(f"Val   Loss: {v_loss:.4f} | Acc B:{v_acc_b:.4f} S:{v_acc_s:.4f}")

        if epoch == 0 and device.type == 'cuda':
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

        if v_loss < (best_loss - EARLY_STOPPING_MIN_DELTA):
            best_loss = v_loss
            epochs_no_impr = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved")
        else:
            epochs_no_impr += 1
            print(f"No improvement for {epochs_no_impr} epoch(s)")

        if EARLY_STOPPING and epoch >= WARMUP_EPOCHS and epochs_no_impr >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Epochs: {epoch+1}/{NUM_EPOCHS} | Best val loss: {best_loss:.4f}")
    print(f"Model saved: {best_model_path}")
    print("="*60)

    # Training history plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(history['train_loss'], label='Train'); axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].set_xlabel('Epoch'); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(history['val_acc_b'], color='green')
    axes[0,1].set_title('Val Binary Accuracy'); axes[0,1].set_xlabel('Epoch'); axes[0,1].grid(True)

    axes[1,0].plot(history['val_acc_s'], color='orange')
    axes[1,0].set_title('Val Subtype Accuracy'); axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(True)

    axes[1,1].plot(history['lr'], color='red')
    axes[1,1].set_title('Learning Rate'); axes[1,1].set_xlabel('Epoch'); axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved: {history_plot_path}")

    # -------------------------------------------------------
    # Final Test Set Evaluation
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    d1_test_p, d1_test_b, d1_test_s = load_dataset1_split('test')
    d2_test_p, d2_test_b, d2_test_s = load_dataset2_split('test')

    test_paths  = d1_test_p + d2_test_p
    test_binary = d1_test_b + d2_test_b
    test_subtype = d1_test_s + d2_test_s
    print(f"Test images: {len(test_paths)} (DS1: {len(d1_test_p)}, DS2: {len(d2_test_p)})")

    model_eval = MultiTaskOralClassifier().to(device)
    model_eval.load_state_dict(torch.load(best_model_path, map_location=device))

    if TTA_ENABLED:
        results = evaluate_model_tta(model_eval, test_paths, test_binary,
                                     test_subtype, device, tta_transforms)
    else:
        test_ds = OralPathologyDataset(test_paths, test_binary, test_subtype, transform=val_transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        results = evaluate_model(model_eval, test_loader, device)

    # Metrics
    acc_b  = accuracy_score(results['targets_binary'],  results['preds_binary'])
    prec_b = precision_score(results['targets_binary'], results['preds_binary'], average='weighted', zero_division=0)
    rec_b  = recall_score(results['targets_binary'],    results['preds_binary'], average='weighted', zero_division=0)
    f1_b   = f1_score(results['targets_binary'],        results['preds_binary'], average='weighted', zero_division=0)

    acc_s  = accuracy_score(results['targets_subtype'],  results['preds_subtype'])
    prec_s = precision_score(results['targets_subtype'], results['preds_subtype'], average='weighted', zero_division=0)
    rec_s  = recall_score(results['targets_subtype'],    results['preds_subtype'], average='weighted', zero_division=0)
    f1_s   = f1_score(results['targets_subtype'],        results['preds_subtype'], average='weighted', zero_division=0)

    print(f"\nBinary  - Acc:{acc_b:.4f} Prec:{prec_b:.4f} Rec:{rec_b:.4f} F1:{f1_b:.4f}")
    print(f"Subtype - Acc:{acc_s:.4f} Prec:{prec_s:.4f} Rec:{rec_s:.4f} F1:{f1_s:.4f}")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_confusion_matrix(results['targets_binary'], results['preds_binary'],
                          ['Benign', 'Malignant'], 'Binary Classification\n(Benign vs Malignant)', axes[0])
    plot_confusion_matrix(results['targets_subtype'], results['preds_subtype'],
                          DS2_CLASSES, 'Subtype Classification\n(7-class)', axes[1])
    plt.tight_layout()
    plt.savefig(confusion_mat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved: {confusion_mat_path}")

    # Save results text
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
        f.write("Per-Class Report (Binary):\n")
        f.write(classification_report(results['targets_binary'], results['preds_binary'],
                                       target_names=['Benign', 'Malignant'], zero_division=0))
        f.write("\nPer-Class Report (Subtype):\n")
        f.write(classification_report(results['targets_subtype'], results['preds_subtype'],
                                       target_names=DS2_CLASSES, zero_division=0))
    print(f"Results saved: {results_file}")


if __name__ == "__main__":
    main()
