import torch
import os
import sys

# Ensure the current directory is in path (should be by default, but good for safety)
sys.path.append(os.getcwd())

def verify_structure():
    print("Verifying Project Structure...")
    
    # 1. Config
    try:
        from configs.config import IMG_SIZE, NUM_SUBTYPES, BACKBONE
        print("✓ Config loaded")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return

    # 2. Utils
    try:
        from utils.common import set_seed
        print("✓ Utils loaded")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return

    # 3. Data
    try:
        from data.dataset import OralPathologyDataset
        from data.transforms import val_transform
        print("✓ Data modules loaded")
    except ImportError as e:
        print(f"✗ Data import failed: {e}")
        return

    # 4. Models
    try:
        from models.architecture import MultiTaskOralClassifier
        from models.loss import MultiTaskLoss
        print("✓ Model modules loaded")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return

    # 5. Engine
    try:
        from engine.trainer import train_one_epoch
        print("✓ Engine loaded")
    except ImportError as e:
        print(f"✗ Engine import failed: {e}")
        return

    # 6. Model Initialization Check
    try:
        model = MultiTaskOralClassifier(num_subtypes=NUM_SUBTYPES, backbone=BACKBONE, dropout=0.5)
        dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        o1, o2 = model(dummy_input)
        print(f"✓ Model forward pass successful. Shapes: {o1.shape}, {o2.shape}")
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return

    print("\nSUCCESS: The folder structure refactoring is complete and verified!")

if __name__ == "__main__":
    verify_structure()
