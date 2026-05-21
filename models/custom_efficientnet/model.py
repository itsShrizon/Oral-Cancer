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

from .attention_hub import AttentionHub, AttentionHubV2


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
                 dropout: float = 0.2, verbose: bool = False,
                 attention_branches=None, hub_version: str = 'v1'):
        """
        attention_branches:
          - None (default): full AttentionHub (BAM + Triplet + KAN). Matches
            the originally proposed model and the existing checkpoint.
          - tuple/list/set with subset of {"bam","triplet","kan"}: build a
            partial AttentionHub for ablation.
          - empty tuple/list (): replace stage4 with the donor's original
            Block-4 (canonical EfficientNetV2-B0 MBConv+SE). This is the
            "no attention" control — exactly the layer the hub replaces.
        """
        super().__init__()
        self.verbose = verbose
        self._num_classes = num_classes
        self.attention_branches = attention_branches
        if hub_version not in ('v1', 'v2'):
            raise ValueError(f"hub_version must be 'v1' or 'v2', got {hub_version!r}")
        self.hub_version = hub_version

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

        # Stage 4: Custom AttentionHub (in=96, out=112) — or no-attention control.
        if hub_version == 'v2':
            # V2: sequential Triplet->EMA cascade with LayerScale gates.
            # Ignores attention_branches (which is a v1-only ablation knob).
            self.stage4 = AttentionHubV2(in_channels=96, out_channels=112)
        elif attention_branches is None:
            self.stage4 = AttentionHub(in_channels=96, out_channels=112)
        elif len(tuple(attention_branches)) == 0:
            # No-attention control: use the donor's original Block-4
            # (MBConv+SE, 96->112) — i.e. the layer the AttentionHub replaces.
            self.stage4 = donor.blocks[4]
        else:
            self.stage4 = AttentionHub(in_channels=96, out_channels=112,
                                       branches=tuple(attention_branches))

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
