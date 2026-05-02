"""
Attention Hub (Stage 4)
-----------------------
Routes Stage 3 output into three parallel attention branches
(BAM, Triplet Attention, KAN), then fuses them back via concatenation
and a 1×1 projection.
"""

import torch
import torch.nn as nn

from .bam import BAM
from .triplet_attention import TripletAttention
from .kan import KANAttention


class AttentionHub(nn.Module):
    """
    Three-branch attention fusion module.

    Args:
        in_channels:  channels coming from Stage 3.
        out_channels: channels expected by Stage 5.
        reduction:    BAM / channel-reduction ratio.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()

        # Per-branch 1×1 channel reduction (saves parameters)
        branch_ch = in_channels // 2

        self.reduce_bam = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.reduce_tri = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.reduce_kan = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )

        # Attention modules operate on reduced channels
        self.bam = BAM(branch_ch, reduction=reduction)
        self.triplet = TripletAttention()
        self.kan = KANAttention(branch_ch)

        # Fusion: concat 3 branches → project to out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.bam(self.reduce_bam(x))
        b2 = self.triplet(self.reduce_tri(x))
        b3 = self.kan(self.reduce_kan(x))

        fused = torch.cat([b1, b2, b3], dim=1)
        return self.fuse(fused)
