"""
CustomEfficientNetV2
--------------------
Parameter-efficient 5-stage CNN based on EfficientNetV2-B0.

Stage layout:
  1. Stem + Block-0 + Block-1  (Fused-MBConv)    → 32 ch
  2. Block-2                   (Fused-MBConv)    → 48 ch
  3. Block-3                   (MBConv)          → 96 ch
  4. AttentionHub (BAM + Triplet + KAN)          → 112 ch  (replaces Block-4)
  5. Block-5                   (MBConv + SE)     → 192 ch
  -- Block-6 / conv_head DROPPED --
  6. GAP → FC classifier

Designed for binary or multiclass classification.
"""

import torch
import torch.nn as nn
import timm

from .attention_hub import AttentionHub


class CustomEfficientNetV2(nn.Module):
    """
    Custom 5-stage EfficientNetV2 with multi-branch attention.

    Args:
        num_classes: output classes (2 for binary, N for multiclass).
                     Set to 0 for feature-extractor mode (returns 192-d vector).
        pretrained:  load ImageNet weights for the retained backbone stages.
        dropout:     dropout probability before the classifier head.
        verbose:     if True, print tensor shapes during forward pass.
    """

    num_features = 192  # exposed for MultiTaskOralClassifier compatibility

    def __init__(self, num_classes: int = 2, pretrained: bool = False,
                 dropout: float = 0.2, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self._num_classes = num_classes

        # --- Load donor backbone ---
        donor = timm.create_model("tf_efficientnetv2_b0",
                                  pretrained=pretrained, num_classes=0)

        # --- Stage 1: Stem (conv_stem + bn1 which includes SiLU) ---
        self.stem = nn.Sequential(donor.conv_stem, donor.bn1)

        # --- Stages from donor.blocks ---
        # Block indices in tf_efficientnetv2_b0:
        #   0: ConvBnAct   (16→32)  — part of stem pathway
        #   1: EdgeResidual (32→32)  — Fused-MBConv
        #   2: EdgeResidual (32→48)  — Fused-MBConv
        #   3: InvertedResidual (48→96) — MBConv
        #   4: InvertedResidual (96→112) — MBConv+SE  ← REPLACED
        #   5: InvertedResidual (112→192) — MBConv+SE ← kept as Stage 5

        # Stage 1 continued: blocks 0-1 (Fused-MBConv, out=32)
        self.stage1 = nn.Sequential(donor.blocks[0], donor.blocks[1])

        # Stage 2: block 2 (Fused-MBConv, out=48)
        self.stage2 = donor.blocks[2]

        # Stage 3: block 3 (MBConv, out=96)
        self.stage3 = donor.blocks[3]

        # Stage 4: Custom AttentionHub (in=96, out=112)
        self.stage4 = AttentionHub(in_channels=96, out_channels=112)

        # Stage 5: block 5 (MBConv+SE, out=192)
        self.stage5 = donor.blocks[5]

        # --- Pooling (always present) ---
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # --- Classifier (skipped when num_classes=0 → feature-extractor mode) ---
        if num_classes > 0:
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Linear(192, num_classes)
        else:
            self.dropout = None
            self.classifier = None

        # Free unused donor weights
        del donor

    def _log(self, tag: str, x: torch.Tensor):
        if self.verbose:
            print(f"  [{tag}] → {list(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.verbose:
            print(f"  [Input]  → {list(x.shape)}")

        x = self.stem(x)
        self._log("Stem", x)

        x = self.stage1(x)
        self._log("Stage 1 — Fused-MBConv", x)

        x = self.stage2(x)
        self._log("Stage 2 — Fused-MBConv", x)

        x = self.stage3(x)
        self._log("Stage 3 — MBConv", x)

        x = self.stage4(x)
        self._log("Stage 4 — AttentionHub", x)

        x = self.stage5(x)
        self._log("Stage 5 — MBConv+SE", x)

        x = self.pool(x)
        x = self.flatten(x)

        if self.classifier is not None:
            x = self.dropout(x)
            x = self.classifier(x)
            self._log("Classifier", x)
        else:
            self._log("Features", x)

        return x
