"""
Attention Hub (Stage 4)
-----------------------
Routes Stage 3 output into selected parallel attention branches
(BAM, Triplet Attention, KAN), then fuses them back via concatenation
and a 1x1 projection.

The set of active branches is controlled by `branches`. The default is
all three, which matches the originally proposed module and is checkpoint
compatible with the existing baseline-recipe run.
"""

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .bam import BAM
from .triplet_attention import TripletAttention
from .kan import KANAttention
from .ema import EMA
from .se import SqueezeExcite


_VALID_BRANCHES: Tuple[str, ...] = ("bam", "triplet", "kan")


class AttentionHub(nn.Module):
    """
    Multi-branch attention fusion module.

    Args:
        in_channels:  channels coming from Stage 3.
        out_channels: channels expected by Stage 5.
        reduction:    BAM / channel-reduction ratio.
        branches:     iterable subset of {"bam","triplet","kan"} controlling
                      which branches are instantiated. Default = all three.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16,
                 branches: Iterable[str] = _VALID_BRANCHES):
        super().__init__()

        branches = tuple(branches)
        for b in branches:
            if b not in _VALID_BRANCHES:
                raise ValueError(f"Unknown branch: {b!r}. Valid: {_VALID_BRANCHES}")
        if len(branches) == 0:
            raise ValueError("AttentionHub requires at least one branch. "
                             "For a no-attention control, replace stage4 with "
                             "the donor block instead of using AttentionHub.")
        # Stable canonical ordering so module names match across runs.
        branches = tuple(b for b in _VALID_BRANCHES if b in branches)
        self.active_branches = branches

        branch_ch = in_channels // 2

        # Branches are constructed conditionally so disabled branches add zero
        # parameters / FLOPs.
        if "bam" in branches:
            self.reduce_bam = nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU(inplace=True),
            )
            self.bam = BAM(branch_ch, reduction=reduction)

        if "triplet" in branches:
            self.reduce_tri = nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU(inplace=True),
            )
            self.triplet = TripletAttention()

        if "kan" in branches:
            self.reduce_kan = nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.SiLU(inplace=True),
            )
            self.kan = KANAttention(branch_ch)

        # Fusion: concat N branches -> project to out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * len(branches), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if "bam" in self.active_branches:
            outs.append(self.bam(self.reduce_bam(x)))
        if "triplet" in self.active_branches:
            outs.append(self.triplet(self.reduce_tri(x)))
        if "kan" in self.active_branches:
            outs.append(self.kan(self.reduce_kan(x)))

        fused = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return self.fuse(fused)


class AttentionHubV2(nn.Module):
    """
    AttentionHub v2 - sequential Triplet -> SE cascade (CBAM-style).

    Design rationale (driven by the v1 ablation + a failed v2-EMA attempt):

      v1 ablation revealed: any module that overlaps Triplet's spatial role
      regresses subtype below the no-attention baseline.
        * `bam_triplet`     -> 98.25 / 98.36   (BAM = spatial+channel)
        * v2 Triplet->EMA   -> 98.60 / 98.36   (EMA = multi-scale spatial)
      The subtype number 98.36 is identical in both failed combos - the
      signature of "two modules fighting for the spatial-attention role".

      Triplet's complement must be a *purely channel* operation:
        * `triplet_kan` (v1, parallel) -> 99.12 / 99.45   <- current best

    AttentionHub-v2 therefore pairs Triplet with **Squeeze-Excitation**:
        * Triplet (Misra et al., WACV 2021) - cross-dim spatial attention.
        * SE (Hu et al., CVPR 2018)         - pure channel recalibration;
          identical role to what KAN plays in `triplet_kan` but with no
          spline-overfitting risk on small data.

    Composition is sequential (Woo et al., CBAM 2018): Triplet handles the
    spatial dimension first, then SE refines channels of the spatially
    attended features. No LayerScale, no per-module residual - both
    attentions are multiplicative gates that preserve input information.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 inner_channels: int = None, se_reduction: int = 16):
        super().__init__()
        ic = inner_channels or in_channels

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, ic, 1, bias=False),
            nn.BatchNorm2d(ic),
            nn.SiLU(inplace=True),
        )

        self.triplet = TripletAttention()
        self.se      = SqueezeExcite(ic, reduction=se_reduction)

        self.expand = nn.Sequential(
            nn.Conv2d(ic, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.triplet(x)
        x = self.se(x)
        x = self.expand(x)
        return x
