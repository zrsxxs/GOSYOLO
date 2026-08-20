"""Standalone wrapper for the OSA-IoU component.

This file intentionally does not reproduce the complete internal Ultralytics
``BboxLoss`` modification. The paper combines the OSA loss with the CIoU loss
using weights of 0.9 and 0.1 inside the detector training objective.
"""

import torch.nn as nn

from .metrics import bbox_iou_osa

__all__ = ("osa_iou_loss", "OSALoss")


def osa_iou_loss(
    pred_bboxes,
    target_bboxes,
    weight=None,
    target_scores_sum=None,
    xywh=False,
    eps=1e-7,
):
    """Return the standalone OSA-IoU loss for aligned boxes."""
    score = bbox_iou_osa(
        pred_bboxes,
        target_bboxes,
        xywh=xywh,
        eps=eps,
    )
    loss = 1.0 - score

    if weight is not None:
        loss = loss * weight
        if target_scores_sum is not None:
            return loss.sum() / target_scores_sum
    return loss.mean()


class OSALoss(nn.Module):
    """Small module wrapper around the standalone OSA-IoU loss."""

    def __init__(self, xywh=False, eps=1e-7):
        super().__init__()
        self.xywh = xywh
        self.eps = eps

    def forward(
        self,
        pred_bboxes,
        target_bboxes,
        weight=None,
        target_scores_sum=None,
    ):
        return osa_iou_loss(
            pred_bboxes,
            target_bboxes,
            weight=weight,
            target_scores_sum=target_scores_sum,
            xywh=self.xywh,
            eps=self.eps,
        )
