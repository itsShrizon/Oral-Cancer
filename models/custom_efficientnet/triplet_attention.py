"""
Triplet Attention
-----------------
Three-branch cross-dimension interaction that captures inter-dependencies
between (C, H), (C, W), and (H, W) with near-zero extra parameters.

Reference: Misra et al., "Rotate to Attend: Convolutional Triplet Attention
Module", WACV 2021.
"""

import torch
import torch.nn as nn


class ZPool(nn.Module):
    """Concatenate max-pool and mean-pool along channel dim → 2 channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x.max(dim=1, keepdim=True).values,
             x.mean(dim=1, keepdim=True)],
            dim=1,
        )


class AttentionGate(nn.Module):
    """ZPool → 7×7 Conv → BN → Sigmoid."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            ZPool(),
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class TripletAttention(nn.Module):
    """
    Triplet Attention Module.

    Three branches:
      1. Permute C↔H, apply gate, permute back  →  captures (C, H) interaction
      2. Permute C↔W, apply gate, permute back  →  captures (C, W) interaction
      3. Identity spatial gate                   →  captures (H, W) interaction

    Outputs are averaged.
    """

    def __init__(self):
        super().__init__()
        self.gate_ch = AttentionGate()  # after C↔H permute
        self.gate_cw = AttentionGate()  # after C↔W permute
        self.gate_hw = AttentionGate()  # spatial (no permute)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: swap C and H  →  (B, H, C, W)
        x_ch = x.permute(0, 2, 1, 3).contiguous()
        x_ch = self.gate_ch(x_ch).permute(0, 2, 1, 3).contiguous()

        # Branch 2: swap C and W  →  (B, W, H, C)
        x_cw = x.permute(0, 3, 2, 1).contiguous()
        x_cw = self.gate_cw(x_cw).permute(0, 3, 2, 1).contiguous()

        # Branch 3: spatial (H, W) — no permutation
        x_hw = self.gate_hw(x)

        return (x_ch + x_cw + x_hw) / 3.0
