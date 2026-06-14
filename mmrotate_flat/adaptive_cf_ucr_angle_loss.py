from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from torch import Tensor

from mmrotate.registry import MODELS


def adaptive_cfucr_angle_loss(
        pred: Tensor,
        target: dict,
        num_bins: int,
        bin_loss_weight: float = 1.0,
        delta_loss_weight: float = 1.0,
        uc_loss_weight: float = 0.01,
        eps: float = 1e-6,
        reduction: str = 'mean',
        avg_factor: Optional[float] = None) -> Tensor:
    """CF-UCR loss with soft-bin targets and geometry-adaptive weighting."""
    assert isinstance(target, dict), \
        'AdaptiveCFUCRAngleLoss expects dict target.'

    logits = pred[..., :num_bins]
    residual_vec = pred[..., num_bins:num_bins + 2]
    angle_weight = target.get(
        'angle_weight',
        pred.new_ones((pred.size(0), ), dtype=pred.dtype)).to(pred.dtype)

    if 'bin_target' in target:
        bin_target = target['bin_target'].to(dtype=pred.dtype)
        cls_loss = -(bin_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        bin_idx = target['bin_idx'].long()
        cls_loss = F.cross_entropy(logits, bin_idx, reduction='none')

    delta_gt = target['delta'].to(dtype=pred.dtype)
    x = residual_vec[..., 0]
    y = residual_vec[..., 1]
    norm = torch.sqrt(x * x + y * y).clamp(min=eps)
    delta_pred = torch.atan2(y / norm, x / norm)

    reg_loss = F.smooth_l1_loss(delta_pred, delta_gt, reduction='none')
    uc_loss = (norm.square() - 1.0).abs()

    loss = (bin_loss_weight * cls_loss * angle_weight
            + delta_loss_weight * reg_loss * angle_weight
            + uc_loss_weight * uc_loss)
    return weight_reduce_loss(
        loss, weight=None, reduction=reduction, avg_factor=avg_factor)


@MODELS.register_module()
class AdaptiveCFUCRAngleLoss(nn.Module):
    """Loss for Adaptive-CF-UCR."""

    def __init__(self,
                 num_bins: int,
                 bin_loss_weight: float = 1.0,
                 delta_loss_weight: float = 1.0,
                 uc_loss_weight: float = 0.01,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.bin_loss_weight = float(bin_loss_weight)
        self.delta_loss_weight = float(delta_loss_weight)
        self.uc_loss_weight = float(uc_loss_weight)
        self.eps = float(eps)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred: Tensor,
                target,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        del weight
        reduction = reduction_override if reduction_override else self.reduction
        loss = adaptive_cfucr_angle_loss(
            pred,
            target,
            num_bins=self.num_bins,
            bin_loss_weight=self.bin_loss_weight,
            delta_loss_weight=self.delta_loss_weight,
            uc_loss_weight=self.uc_loss_weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)
        return self.loss_weight * loss
