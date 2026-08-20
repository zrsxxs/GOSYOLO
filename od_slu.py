"""Orthogonal Spatial-decoupled Lookup Unit used in GOS-YOLO."""

import torch.nn as nn

__all__ = ("OD_SLU",)


class OD_SLU(nn.Module):
    """Estimate horizontal and vertical weights through disjoint paths."""

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        hidden = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1_h = nn.Conv2d(inp, hidden, kernel_size=1)
        self.bn1_h = nn.BatchNorm2d(hidden)
        self.act_h = nn.Hardswish()
        self.conv2_h = nn.Conv2d(hidden, oup, kernel_size=1)

        self.conv1_w = nn.Conv2d(inp, hidden, kernel_size=1)
        self.bn1_w = nn.BatchNorm2d(hidden)
        self.act_w = nn.Hardswish()
        self.conv2_w = nn.Conv2d(hidden, oup, kernel_size=1)

    def forward(self, x):
        row_descriptor = self.pool_h(x)
        column_descriptor = self.pool_w(x).permute(0, 1, 3, 2)

        row_weight = self.conv1_h(row_descriptor)
        row_weight = self.act_h(self.bn1_h(row_weight))
        row_weight = self.conv2_h(row_weight).sigmoid()

        column_weight = self.conv1_w(column_descriptor)
        column_weight = self.act_w(self.bn1_w(column_weight))
        column_weight = self.conv2_w(column_weight).sigmoid()
        column_weight = column_weight.permute(0, 1, 3, 2)

        return x * row_weight * column_weight

