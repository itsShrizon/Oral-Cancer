"""
Squeeze-and-Excitation channel attention.

Reference: Hu, Shen, Sun. "Squeeze-and-Excitation Networks." CVPR 2018.

Used as the second module in AttentionHub v2's sequential cascade
(Triplet -> SE). Role separation:
  - Triplet handles spatial / cross-dimensional attention.
  - SE handles pure channel-wise recalibration.
This pairing avoids the role-overlap that caused BAM+Triplet and EMA+Triplet
to regress below the no-attention baseline on this dataset.
"""

import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, min_hidden: int = 4):
        super().__init__()
        hidden = max(channels // reduction, min_hidden)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.gap(x))
