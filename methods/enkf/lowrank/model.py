"""A compact covariance-only U-Net for the frozen PedPred3 forecast.

The network sees ``[x_t, mean_t1, mean_t1 - x_t, static_mask]`` and predicts a
factor and positive diagonal for ``B = U U^T + diag(d)``. It never sees the
future observation, so the result remains a background (prior) covariance.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Per-channel state scales measured on training data (see methods/dincae/runs/state_stats.log).
STATE_SCALE = (0.19491, 0.84916, 0.33664, 0.14897)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        groups = math.gcd(8, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class CovarianceUNet(nn.Module):
    """Two-head U-Net producing low-rank factors and diagonal variances."""

    def __init__(self, channels=4, rank=16, width=32, diagonal_floor=1e-5,
                 diagonal_floor_fraction=0.0):
        super().__init__()
        self.channels, self.rank = channels, rank
        self.diagonal_floor = float(diagonal_floor)
        self.diagonal_floor_fraction = float(diagonal_floor_fraction)
        self.register_buffer("state_scale", torch.tensor(STATE_SCALE).view(1, channels, 1, 1))

        self.enc0 = ConvBlock(3 * channels + 1, width)
        self.enc1 = ConvBlock(width, 2 * width)
        self.bottom = ConvBlock(2 * width, 4 * width)
        self.dec1 = ConvBlock(6 * width, 2 * width)
        self.dec0 = ConvBlock(3 * width, width)
        self.factor_head = nn.Conv2d(width, rank * channels, 1)
        self.diagonal_head = nn.Conv2d(width, channels, 1)

        # Begin close to a diagonal filter, but not exactly at U=0: because B contains
        # U U^T, an exactly-zero factor has exactly-zero gradient and can never learn.
        nn.init.normal_(self.factor_head.weight, std=1e-3)
        nn.init.zeros_(self.factor_head.bias)
        nn.init.zeros_(self.diagonal_head.weight)
        nn.init.constant_(self.diagonal_head.bias, math.log(math.expm1(1.0)))

    def forward(self, x_t, mean_t1, static_mask=None):
        if x_t.ndim == 5:
            x_t = x_t[:, 0]
        if mean_t1.ndim == 5:
            mean_t1 = mean_t1[:, 0]
        if x_t.shape != mean_t1.shape or x_t.shape[1] != self.channels:
            raise ValueError("x_t and mean_t1 must both be (B,C,H,W)")
        scale = self.state_scale.to(dtype=x_t.dtype)
        if static_mask is None:
            static_mask = torch.ones((len(x_t), 1, *x_t.shape[-2:]),
                                     dtype=x_t.dtype, device=x_t.device)
        elif static_mask.ndim == 2:
            static_mask = static_mask[None, None].expand(len(x_t), 1, -1, -1)
        elif static_mask.ndim == 3:
            static_mask = static_mask[:, None]
        if static_mask.shape != (len(x_t), 1, *x_t.shape[-2:]):
            raise ValueError("static_mask must be (H,W), (B,H,W), or (B,1,H,W)")
        inp = torch.cat((x_t / scale, mean_t1 / scale, (mean_t1 - x_t) / scale,
                         static_mask.to(x_t.dtype)), dim=1)

        e0 = self.enc0(inp)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        z = self.bottom(F.avg_pool2d(e1, 2))
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(torch.cat((z, e1), dim=1))
        z = F.interpolate(z, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec0(torch.cat((z, e0), dim=1))

        b, _, h, w = z.shape
        raw_u = self.factor_head(z).reshape(b, self.rank, self.channels, h, w)
        # Physical units, with rank-independent initial/typical total variance.
        factor = torch.tanh(raw_u) * scale[:, None] / math.sqrt(self.rank)
        diagonal = ((F.softplus(self.diagonal_head(z)) + self.diagonal_floor_fraction)
                    * scale.square() + self.diagonal_floor)
        return factor, diagonal
