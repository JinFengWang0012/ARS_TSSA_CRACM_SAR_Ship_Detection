# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss

from mmrotate.registry import MODELS


def csl_ucr_loss(pred,
                 target,
                 weight=None,
                 gamma=2.0,
                 alpha=0.25,
                 angle_version='le90',
                 uc_weight=0.1,
                 prob_thr=0.0,
                 reduction='mean',
                 avg_factor=None,
                 eps=1e-6):
    """CSL + UCR-style unit-circle constrained loss.

    Args:
        pred (Tensor): Predicted logits with shape (N, K).
        target (Tensor): CSL soft labels with shape (N, K).
        weight (Tensor, optional): Sample-wise weights.
        gamma (float): Focal loss gamma.
        alpha (float): Focal loss alpha.
        angle_version (str): 'le90', 'oc', or 'le135'.
        uc_weight (float): Weight for the unit-circle regularization term.
        prob_thr (float): Only apply UC term on samples whose max prob
            is greater than this threshold. Set 0 to always apply.
        reduction (str): 'none', 'mean' or 'sum'.
        avg_factor (int, optional): Average factor.
        eps (float): Numerical epsilon.
    """

    # -----------------------------
    # 1) 基础：Smooth Focal（跟 SmoothFocalLoss 一致风格）
    # -----------------------------
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    # pt = p_t in focal
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)

    loss_raw = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight  # (N, K)

    loss_ce = weight_reduce_loss(
        loss_raw, weight, reduction=reduction, avg_factor=avg_factor)

    # -----------------------------
    # 2) UCR 式单位圆约束：作用在预测分布上
    # -----------------------------
    N, K = pred.shape
    device = pred.device

    # logits -> 概率分布（归一化成一圈）
    prob = pred_sigmoid / pred_sigmoid.sum(
        dim=-1, keepdim=True).clamp_min(eps)  # (N, K)

    # angle_version 对应的角度范围
    if angle_version == 'le90':
        angle_range = 180.0
        angle_offset = 90.0
    elif angle_version == 'oc':
        angle_range = 90.0
        angle_offset = 0.0
    elif angle_version == 'le135':
        angle_range = 180.0
        angle_offset = 45.0
    else:
        # default: le90
        angle_range = 180.0
        angle_offset = 90.0

    # 计算每个 bin 的中心角（弧度）
    omega = angle_range / float(K)
    idx = torch.arange(K, device=device, dtype=pred.dtype)
    centers_deg = (idx + 0.5) * omega - angle_offset
    centers_rad = centers_deg * (math.pi / 180.0)  # (K,)

    cos_theta = torch.cos(centers_rad).unsqueeze(0)  # (1, K)
    sin_theta = torch.sin(centers_rad).unsqueeze(0)  # (1, K)

    # 在单位圆上的“期望点”
    mx = (prob * cos_theta).sum(dim=-1)  # (N,)
    my = (prob * sin_theta).sum(dim=-1)  # (N,)

    # 理想情况：mx^2 + my^2 ≈ 1
    uc = (mx ** 2 + my ** 2 - 1.0).abs()  # (N,)

    if prob_thr > 0:
        max_prob, _ = prob.max(dim=-1)  # (N,)
        mask = (max_prob > prob_thr).float()
        valid = mask.sum()
        if valid > 0:
            uc = (uc * mask).sum() / valid
        else:
            uc = uc.mean()
    else:
        uc = uc.mean()

    loss_uc = uc_weight * uc

    return loss_ce + loss_uc


@MODELS.register_module()
class CSLUCRLoss(nn.Module):
    """CSL loss with UCR-style unit-circle regularization."""

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 angle_version='le90',
                 uc_weight=0.1,
                 prob_thr=0.0,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.angle_version = angle_version
        self.uc_weight = uc_weight
        self.prob_thr = prob_thr
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, K).
            target (Tensor): CSL soft labels with shape (N, K).
            weight (Tensor, optional): Sample-wise weights.
            avg_factor (int, optional): Average factor used to average loss.
            reduction_override (str, optional): Override reduction method.

        Returns:
            Tensor: loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        loss = csl_ucr_loss(
            pred,
            target,
            weight=weight,
            gamma=self.gamma,
            alpha=self.alpha,
            angle_version=self.angle_version,
            uc_weight=self.uc_weight,
            prob_thr=self.prob_thr,
            reduction=reduction,
            avg_factor=avg_factor)

        return self.loss_weight * loss
