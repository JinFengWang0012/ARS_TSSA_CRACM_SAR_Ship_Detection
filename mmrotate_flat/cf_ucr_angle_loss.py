# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from torch import Tensor

from mmrotate.registry import MODELS


def cfucr_angle_loss(pred: Tensor,
                     target: dict,
                     num_bins: int,
                     bin_loss_weight: float = 1.0,
                     delta_loss_weight: float = 1.0,
                     uc_loss_weight: float = 0.01,
                     eps: float = 1e-6,
                     reduction: str = 'mean',
                     avg_factor: Optional[float] = None) -> Tensor:
    """Coarse-to-Fine UCR angle loss.

    Args:
        pred (Tensor): 预测的角度编码，形状 (N, num_bins + 2)。
        target (dict): 由 CFUCRAngleCoder.encode() 返回的字典：
            - bin_idx: (N,) 每个样本所属的方向簇索引。
            - delta: (N,) 簇内残差角度（弧度）。
        num_bins (int): 方向簇数量。
        bin_loss_weight (float): 粗方向分类损失权重。
        delta_loss_weight (float): 残差回归损失权重。
        uc_loss_weight (float): 单位圆正则项权重。
        eps (float): 数值稳定项。
        reduction (str): {'none', 'mean', 'sum'}。
        avg_factor (float, optional): 损失归一化因子。

    Returns:
        Tensor: 标量损失。
    """
    assert isinstance(target, dict), 'CFUCRAngleLoss expects dict target.'
    bin_idx: Tensor = target['bin_idx'].long()
    delta_gt: Tensor = target['delta']

    # 拆分预测向量
    logits = pred[..., :num_bins]        # (N, K)
    residual_vec = pred[..., num_bins:num_bins + 2]  # (N, 2)

    # (1) 粗方向分类：标准交叉熵
    cls_loss = F.cross_entropy(
        logits, bin_idx, reduction='none')  # (N,)

    # (2) 细残差：将 2D 向量归一化到单位圆，并用 atan2 解码 δ
    x = residual_vec[..., 0]
    y = residual_vec[..., 1]
    norm = torch.sqrt(x * x + y * y).clamp(min=eps)
    x_n = x / norm
    y_n = y / norm
    delta_pred = torch.atan2(y_n, x_n)

    reg_loss = F.smooth_l1_loss(
        delta_pred, delta_gt, reduction='none')  # (N,)

    # (3) Unit-circle 正则：鼓励 ||(x, y)|| -> 1
    uc_loss = (norm.pow(2.0) - 1.0).abs()

    # 组合三个部分
    loss = (bin_loss_weight * cls_loss
            + delta_loss_weight * reg_loss
            + uc_loss_weight * uc_loss)

    loss = weight_reduce_loss(
        loss, weight=None, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class CFUCRAngleLoss(nn.Module):
    """Angle loss for CF-UCR.

    组合：
    - 粗方向分类交叉熵；
    - 簇内残差 Smooth L1 回归；
    - 单位圆范数约束（UCR 风格正则）。
    """

    def __init__(self,
                 num_bins: int,
                 bin_loss_weight: float = 1.0,
                 delta_loss_weight: float = 1.0,
                 uc_loss_weight: float = 0.01,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert num_bins > 1
        self.num_bins = int(num_bins)
        self.bin_loss_weight = float(bin_loss_weight)
        self.delta_loss_weight = float(delta_loss_weight)
        self.uc_loss_weight = float(uc_loss_weight)
        self.eps = float(eps)
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred: Tensor,
                target,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """前向计算。

        Args:
            pred (Tensor): (N, num_bins + 2) 的预测编码。
            target (dict): CFUCRAngleCoder.encode() 的输出。
            weight (Tensor, optional): 未使用，仅与 mmdet 接口对齐。
            avg_factor (float, optional): 损失归一化因子。
            reduction_override (str, optional): 临时覆盖 reduction。
        """
        reduction = reduction_override if reduction_override else self.reduction
        loss = cfucr_angle_loss(
            pred,
            target,
            num_bins=self.num_bins,
            bin_loss_weight=self.bin_loss_weight,
            delta_loss_weight=self.delta_loss_weight,
            uc_loss_weight=self.uc_loss_weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * loss
