"""
Test script for CustomEfficientNetV2.
Verifies forward-pass shapes and prints parameter count.

Usage:
    cd /workspace/Oral-Cancer && python -m tests.test_custom_efficientnet
"""

import torch
from models.custom_efficientnet import CustomEfficientNetV2


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model(num_classes: int, label: str):
    print(f"\n{'=' * 60}")
    print(f" {label}  (num_classes={num_classes})")
    print('=' * 60)

    model = CustomEfficientNetV2(num_classes=num_classes, pretrained=False,
                                  verbose=True)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, num_classes), (
        f"Expected output shape (1, {num_classes}), got {out.shape}"
    )
    print(f"\n  ✓ Output shape OK: {list(out.shape)}")

    total, trainable = count_params(model)
    print(f"  ✓ Total params:     {total:,}")
    print(f"  ✓ Trainable params: {trainable:,}")


if __name__ == "__main__":
    test_model(num_classes=2, label="Binary Classification")
    test_model(num_classes=7, label="Multiclass Classification (7-class)")
    print(f"\n{'=' * 60}")
    print(" All tests passed ✓")
    print('=' * 60)
