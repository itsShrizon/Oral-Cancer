"""
Bottleneck Attention Module (BAM)
---------------------------------
Combines channel attention (GAP → FC bottleneck) and spatial attention
(1×1 → dilated 3×3 convs) into a single sigmoid gate.

Reference: Park et al., "BAM: Bottleneck Attention Module", BMVC 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """Channel attention via global-average-pool → FC bottleneck → FC expand."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        gap = F.adaptive_avg_pool2d(x, 1).view(b, c)  # (B, C)
        att = self.mlp(gap).view(b, c, 1, 1)           # (B, C, 1, 1)
        return att.expand_as(x)


class SpatialGate(nn.Module):
    """Spatial attention via 1×1 reduction → two dilated 3×3 convs → 1×1 expand."""

    def __init__(self, channels: int, reduction: int = 16, dilation: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),  # single-channel spatial map
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.conv(x)  # (B, 1, H, W)
        return att.expand_as(x)


class BAM(nn.Module):
    """
    Bottleneck Attention Module.

    Args:
        channels:  number of input (and output) channels.
        reduction: bottleneck reduction ratio (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = torch.sigmoid(self.channel_gate(x) + self.spatial_gate(x))
        return x * att
