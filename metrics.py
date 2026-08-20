"""Minimal IoU and OSA-IoU calculations for the public implementation.

This file retains the geometric operations required by OSA-IoU. Other IoU-loss
variants from the internal experiment code are intentionally omitted.
"""

import torch

__all__ = ("bbox_iou_osa",)


def bbox_iou_osa(
    box1,
    box2,
    xywh=True,
    eps=1e-7,
    return_plain_iou=False,
):
    """Calculate OSA-IoU for aligned boxes.

    The second argument is treated as the target box because the cross weights
    depend on target width and height. Inputs use ``xywh`` format when ``xywh``
    is true and ``xyxy`` format otherwise.
    """
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, dim=-1)
        x2, y2, w2, h2 = box2.chunk(4, dim=-1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
    intersection = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - intersection + eps
    plain_iou = intersection / union

    enclosing_w = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    enclosing_h = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    center_error_x = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
    ) / 4
    center_error_y = (
        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4

    axis_error_x = (
        center_error_x + (w2 - w1) ** 2
    ) / (enclosing_w**2 + eps)
    axis_error_y = (
        center_error_y + (h2 - h1) ** 2
    ) / (enclosing_h**2 + eps)

    weight_x = h2 / (w2 + h2 + eps)
    weight_y = w2 / (w2 + h2 + eps)
    osa_distance = weight_x * axis_error_x + weight_y * axis_error_y
    osa_penalty = 1.0 - torch.exp(-osa_distance)
    osa_iou = plain_iou - osa_penalty

    if return_plain_iou:
        return osa_iou, plain_iou
    return osa_iou

