# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from torch import Tensor

from mmrotate.registry import MODELS


def acm_consistency_loss(pred: Tensor,
                         target: Tensor,
                         weight: Optional[Tensor] = None,
                         beta: float = 1.0 / 9.0,
                         reg_weight: float = 1.0,
                         unit_circle_weight: float = 0.02,
                         phase_consistency_weight: float = 0.05,
                         reduction: str = 'mean',
                         avg_factor: Optional[float] = None) -> Tensor:
    """ACM angle loss with light consistency regularization.

    The head still predicts the standard ACM code
    ``[cos(2a), sin(2a), cos(4a), sin(4a)]``. Besides encoded-angle
    regression, this loss softly encourages each frequency pair to stay on the
    unit circle and keeps the two frequency phases mutually consistent.
    """
    if pred.numel() == 0:
        return pred.sum()

    assert pred.size(-1) == 4 and target.size(-1) == 4, \
        'ACMConsistencyLoss expects 4-D ACM codes.'

    reg = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    reg = weight_reduce_loss(reg, weight, reduction, avg_factor)

    cos2, sin2, cos4, sin4 = pred.unbind(dim=-1)
    norm2 = torch.sqrt(cos2.square() + sin2.square()).clamp(min=1e-6)
    norm4 = torch.sqrt(cos4.square() + sin4.square()).clamp(min=1e-6)

    unit = 0.5 * ((norm2 - 1.0).square() + (norm4 - 1.0).square())
    unit = weight_reduce_loss(unit, weight, reduction, avg_factor)

    phase2 = torch.atan2(sin2 / norm2, cos2 / norm2)
    phase4 = torch.atan2(sin4 / norm4, cos4 / norm4)
    phase = 1.0 - torch.cos(2.0 * phase2 - phase4)
    phase = weight_reduce_loss(phase, weight, reduction, avg_factor)

    return (reg_weight * reg + unit_circle_weight * unit +
            phase_consistency_weight * phase)


@MODELS.register_module()
class ACMConsistencyLoss(nn.Module):
    """Lightweight ACM loss for unchanged RotatedFCOSHead."""

    def __init__(self,
                 beta: float = 1.0 / 9.0,
                 reg_weight: float = 1.0,
                 unit_circle_weight: float = 0.02,
                 phase_consistency_weight: float = 0.05,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.beta = beta
        self.reg_weight = reg_weight
        self.unit_circle_weight = unit_circle_weight
        self.phase_consistency_weight = phase_consistency_weight
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = acm_consistency_loss(
            pred,
            target,
            weight=weight,
            beta=self.beta,
            reg_weight=self.reg_weight,
            unit_circle_weight=self.unit_circle_weight,
            phase_consistency_weight=self.phase_consistency_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss
