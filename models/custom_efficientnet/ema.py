"""
Efficient Multi-scale Attention (EMA)
-------------------------------------
Reference: Ouyang, Yan, Zhao, Zheng, He, Yang. "Efficient Multi-Scale Attention
Module with Cross-Spatial Learning". ICASSP 2023.

Groups channels, then combines (1) parallel 1x1 path that captures positional
information along H and W via strip pooling and (2) a 3x3 path for local
context. The two paths are combined with cross-spatial softmax weighting,
producing a per-pixel attention map applied to the input features.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """
    Args:
        channels: input/output channel count.
        factor:   number of groups (channels must be divisible by `factor`).
                  Paper recommends factor=8 or factor=16.
    """

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        if channels % factor != 0:
            raise ValueError(
                f"EMA: channels ({channels}) must be divisible by factor ({factor})."
            )
        self.groups = factor
        gc = channels // factor

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(gc, gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, -1, h, w)            # (b*g, c/g, H, W)

        x_h = self.pool_h(gx)                                # (b*g, c/g, H, 1)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)            # (b*g, c/g, W, 1)
        hw  = self.conv1x1(torch.cat([x_h, x_w], dim=2))     # (b*g, c/g, H+W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)            # split back
        x_w = x_w.permute(0, 1, 3, 2)                        # (b*g, c/g, 1, W)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.sigmoid())     # 1x1 path
        x2 = self.conv3x3(gx)                                # 3x3 path

        # Cross-spatial reweighting
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, -1, h * w)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, -1, h * w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)) \
                    .reshape(b * self.groups, 1, h, w)

        return (gx * weights.sigmoid()).reshape(b, c, h, w)
