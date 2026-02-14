
import torch
import torch.nn as nn
import timm
from models.architecture import MultiTaskOralClassifier, BACKBONE_MAP
import sys

def test_backbones():
    print(f"Testing {len(BACKBONE_MAP)} backbones...")
    
    success = True
    
    for name in BACKBONE_MAP.keys():
        print(f"\n--- Testing BACKBONE='{name}' ---")
        
        try:
            # Pass name directly
            model = MultiTaskOralClassifier(backbone=name)
            model.eval()
            
            # Dummy input
            x = torch.randn(2, 3, 224, 224)
            
            with torch.no_grad():
                out_bin, out_sub = model(x)
            
            print(f"Output shapes: Binary={out_bin.shape}, Subtype={out_sub.shape}")
            
            if out_bin.shape != (2, 2):
                print(f"❌ Binary output shape mismatch: {out_bin.shape} != (2, 2)")
                success = False
            if out_sub.shape != (2, 7): # assuming NUM_SUBTYPES=7
                print(f"❌ Subtype output shape mismatch: {out_sub.shape} != (2, 7)")
                success = False
                
            print(f"✓ {name} passed")
            
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            success = False

    if success:
        print("\n✅ All backbones verified successfully!")
    else:
        print("\n❌ Some backbones failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    test_backbones()
