"""Compact paper-style U-Net that predicts 16 local covariance maps."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LocalCovarianceUNet(nn.Module):
    """Depth-2, width-32 U-Net, matching the scale of Lu's architecture."""
    def __init__(self, in_channels=5, width=32, out_channels=16):
        super().__init__()
        self.enc0 = DoubleConv(in_channels, width)
        self.enc1 = DoubleConv(width, 2 * width)
        self.bottom = DoubleConv(2 * width, 4 * width)
        self.up1 = nn.Conv2d(4 * width, 2 * width, 1)
        self.dec1 = DoubleConv(4 * width, 2 * width)
        self.up0 = nn.Conv2d(2 * width, width, 1)
        self.dec0 = DoubleConv(2 * width, width)
        self.head = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(F.max_pool2d(e0, 2))
        z = self.bottom(F.max_pool2d(e1, 2))
        z = F.interpolate(self.up1(z), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(torch.cat((z, e1), dim=1))
        z = F.interpolate(self.up0(z), size=e0.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(torch.cat((z, e0), dim=1)))
