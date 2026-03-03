"""
KAN Attention Block
-------------------
Kolmogorov-Arnold Network–inspired channel recalibration.

Instead of a plain FC layer, each channel is transformed by a learnable
B-spline basis expansion, giving the network per-channel non-linear
activation control — more expressive than SE with comparable parameter cost.

Applied as: GAP → KAN transform → Sigmoid → channel-wise gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineActivation(nn.Module):
    """
    Learnable 1-D B-spline activation replaces fixed ReLU/SiLU.

    For each of `channels` units the output is a weighted sum of
    `num_bases` B-spline basis functions evaluated on the input scalar,
    plus a residual SiLU connection for stable gradients.

    Args:
        channels:   number of independent activations (one per channel).
        num_bases:  number of spline basis functions (controls expressivity).
        grid_range: symmetric interval for the spline grid.
    """

    def __init__(self, channels: int, num_bases: int = 5,
                 grid_range: float = 3.0):
        super().__init__()
        self.channels = channels
        self.num_bases = num_bases

        # Grid knots evenly spaced in [-grid_range, grid_range]
        grid = torch.linspace(-grid_range, grid_range, num_bases)
        self.register_buffer("grid", grid)  # (num_bases,)

        # Learnable spline coefficients: one set per channel
        self.coeffs = nn.Parameter(torch.randn(channels, num_bases) * 0.1)

        # Grid spacing (for basis width)
        self.h = (2 * grid_range) / (num_bases - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C) — one scalar per channel per sample.
        Returns:
            (B, C) — recalibrated values.
        """
        # Compute distance from each knot: (B, C, num_bases)
        x_expand = x.unsqueeze(-1)               # (B, C, 1)
        bases = ((x_expand - self.grid) / self.h)  # (B, C, num_bases)
        # RBF-style soft basis (smoother than hard B-spline, still learnable)
        bases = torch.exp(-0.5 * bases.pow(2))     # Gaussian basis

        # Weighted sum over bases: (B, C)
        spline_out = (bases * self.coeffs).sum(dim=-1)

        # Residual SiLU for gradient stability
        return spline_out + F.silu(x)


class KANAttention(nn.Module):
    """
    KAN-based channel attention block.

    Flow:  GAP → BSplineActivation (per-channel) → Sigmoid → gate input.

    Args:
        channels: number of input/output channels.
        num_bases: spline basis count (default 5 — keeps params low).
    """

    def __init__(self, channels: int, num_bases: int = 5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.spline = BSplineActivation(channels, num_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.gap(x).view(b, c)        # (B, C)
        s = torch.sigmoid(self.spline(s))  # (B, C)
        return x * s.view(b, c, 1, 1)
