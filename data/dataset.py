import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from configs.config import (
    DS1_ORIGINAL_BENIGN, DS1_ORIGINAL_MALIGNANT,
    DS2_TRAINING, DS2_VALIDATION, DS2_TESTING,
    DS2_CLASSES, MALIGNANT_SUBTYPES, BATCH_SIZE, NUM_WORKERS
)

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
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels_binary[idx], self.labels_subtype[idx]

def get_image_files(folder):
    if not os.path.exists(folder): return []
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
    
    # First split: separate test set (held out completely)
    temp_paths, test_paths, temp_bin, test_bin, temp_sub, test_sub = train_test_split(
        all_paths, all_binary, all_subtype, test_size=test_size, random_state=random_state, stratify=all_binary
    )
    
    # Second split: divide remaining into train and validation
    val_size_adj = val_size / (1 - test_size)
    train_paths, val_paths, train_bin, val_bin, train_sub, val_sub = train_test_split(
        temp_paths, temp_bin, temp_sub, test_size=val_size_adj, random_state=random_state, stratify=temp_bin
    )
    
    if split == 'train': return train_paths, train_bin, train_sub
    elif split == 'val': return val_paths, val_bin, val_sub
    else: return test_paths, test_bin, test_sub

def load_dataset2_split(split='train', test_size=0.20, val_size=0.20, random_state=42):
    """
    Load Dataset 2 with proper train/val/test split.
    MERGES Training + Validation folders, then splits properly to avoid
    the suspicious pre-made Testing folder with identical distributions.
    """
    image_paths, labels_binary, labels_subtype = [], [], []
    
    # Merge Training + Validation folders (ignore the suspicious Testing folder)
    for base_path in [DS2_TRAINING, DS2_VALIDATION]:
        for idx, subtype in enumerate(DS2_CLASSES):
            subtype_path = os.path.join(base_path, subtype)
            imgs = get_image_files(subtype_path)
            image_paths.extend(imgs)
            labels_subtype.extend([idx] * len(imgs))
            labels_binary.extend([1 if subtype in MALIGNANT_SUBTYPES else 0] * len(imgs))
    
    # Now split this merged data properly
    # First split: separate test set (held out completely)
    temp_paths, test_paths, temp_bin, test_bin, temp_sub, test_sub = train_test_split(
        image_paths, labels_binary, labels_subtype, 
        test_size=test_size, random_state=random_state, stratify=labels_subtype
    )
    
    # Second split: divide remaining into train and validation
    val_size_adj = val_size / (1 - test_size)
    train_paths, val_paths, train_bin, val_bin, train_sub, val_sub = train_test_split(
        temp_paths, temp_bin, temp_sub, 
        test_size=val_size_adj, random_state=random_state, stratify=temp_sub
    )
    
    if split == 'train': 
        return train_paths, train_bin, train_sub
    elif split == 'val': 
        return val_paths, val_bin, val_sub
    else: 
        return test_paths, test_bin, test_sub
