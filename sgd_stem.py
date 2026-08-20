"""Structure-Gradient Dual-stream Stem used in GOS-YOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("Sobel", "SGD_Stem")


class Sobel(nn.Module):
    """Apply fixed horizontal and vertical Sobel operators."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x):
        intensity = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(intensity, self.sobel_x.type_as(x), padding=1)
        grad_y = F.conv2d(intensity, self.sobel_y.type_as(x), padding=1)
        return torch.cat((grad_x, grad_y), dim=1)


class SGD_Stem(nn.Module):
    """Modulate the input with a fixed-gradient branch before downsampling."""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.sobel = Sobel()
        self.attn_gen = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )
        self.dw_conv = nn.Conv2d(
            c1,
            c1,
            kernel_size=k,
            stride=s,
            padding=k // 2,
            groups=c1,
            bias=False,
        )
        self.pw_conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        gradient = self.sobel(x)
        attention = self.attn_gen(gradient)
        guided = x * (1.0 + attention)
        output = self.dw_conv(guided)
        output = self.pw_conv(output)
        return self.act(self.bn(output))

