"""Gradient-Calibrated Alignment Fusion used in GOS-YOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("GCAF",)


class ConvBNAct(nn.Module):
    """Minimal convolution, batch-normalization, and SiLU block."""

    def __init__(self, c1, c2, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GCAF(nn.Module):
    """Calibrate an upsampled deep feature with a shallow spatial mask."""

    def __init__(self, c1, c2, c_out):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        self.align_conv = ConvBNAct(c1, c1, kernel_size=3)
        self.calib_weight = nn.Sequential(
            nn.Conv2d(c2, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(c1 + c2, c_out, kernel_size=1)

    def forward(self, x):
        deep, shallow = x
        deep = self.align_conv(self.upsample(deep))
        calibration = self.calib_weight(shallow)

        if deep.shape[2:] != calibration.shape[2:]:
            deep = F.interpolate(
                deep,
                size=calibration.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        calibrated = deep * calibration
        return self.fuse(torch.cat((shallow, calibrated), dim=1))

