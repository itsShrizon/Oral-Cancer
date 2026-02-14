import os
import random
import numpy as np
import torch
from glob import glob
from configs.config import SEED

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device():
    """Get the current device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def check_paths_exist(paths_list):
    """Verify that all paths in the list exist."""
    all_exist = True
    print("\nChecking paths...")
    for path in paths_list:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {path}")
        if not exists:
            all_exist = False
    return all_exist

def count_images_in_folder(folder):
    """Count image files in a folder."""
    if not os.path.exists(folder):
        return 0
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    count = 0
    for ext in extensions:
        count += len(glob(os.path.join(folder, ext)))
        count += len(glob(os.path.join(folder, ext.upper())))
    return count
