# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox.transforms import norm_angle


@TASK_UTILS.register_module()
class CFUCRAngleCoder(BaseBBoxCoder):
    r"""Coarse-to-Fine Unit Circle Resolver (CF-UCR) angle coder.

    这个角度编码器实现了“粗方向分类 + 细粒度残差回归”的思想：
    - 将角度空间划分为若干方向簇（num_bins），用分类预测所属簇；
    - 在簇内仅回归小范围残差，并用 2D 单位圆 (cos δ, sin δ) 表示残差。

    编码输出的通道布局为：
        [ coarse_logits(num_bins) | residual_vector(2) ]

    其中 decode() 会将 coarse_logits 经过 softmax 得到簇中心角的加权和，
    再叠加单位圆解码得到的残差 δ，输出最终角度。
    """

    def __init__(self,
                 num_bins: int = 8,
                 angle_version: str = 'le90',
                 max_residual_ratio: float = 0.5,
                 eps: float = 1e-6) -> None:
        """
        Args:
            num_bins (int): 角度空间划分的簇数 K。
            angle_version (str): 角度表示方式（与数据集/box 一致），如 'le90'。
            max_residual_ratio (float): 残差裁剪范围，相对于每个 bin 宽度的比例。
            eps (float): 数值稳定项。
        """
        super().__init__()
        assert num_bins > 1
        self.num_bins = int(num_bins)
        self.angle_version = angle_version
        self.eps = eps

        # 根据 angle_version 决定角度范围
        if angle_version == 'le90':
            min_angle = -math.pi / 2.0
            max_angle = math.pi / 2.0
        elif angle_version == 'le135':
            # 对应 norm_angle 中的 (angle + pi/4) % pi - pi/4
            min_angle = -math.pi / 4.0
            max_angle = 3.0 * math.pi / 4.0
        elif angle_version == 'r360':
            min_angle = -math.pi
            max_angle = math.pi
        else:
            # 其它角度版本依然可以使用，只要与 norm_angle 一致
            # 这里默认按 [-pi, pi] 处理
            min_angle = -math.pi
            max_angle = math.pi

        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.total_range = self.max_angle - self.min_angle
        self.bin_size = self.total_range / float(self.num_bins)

        # 残差允许的最大幅度（通常远小于 bin 宽度）
        self.max_residual = float(max_residual_ratio) * self.bin_size

        # 角度分支输出维度 = coarse logits + 残差二维向量
        self.encode_size = self.num_bins + 2

    # ------------------------ helper ------------------------ #
    def _angle_to_bin(self, angle: Tensor) -> Tensor:
        """将归一化后的角度映射到 [0, num_bins-1] 的整型 bin index。"""
        # 映射到 [0, num_bins)
        pos = (angle - self.min_angle) / self.bin_size
        idx = torch.floor(pos).to(dtype=torch.long)
        return idx.clamp(min=0, max=self.num_bins - 1)

    def _bin_centers(self, device, dtype) -> Tensor:
        """返回各个方向簇的中心角度 (num_bins,)。"""
        start = self.min_angle + 0.5 * self.bin_size
        end = self.max_angle - 0.5 * self.bin_size
        return torch.linspace(
            start,
            end,
            steps=self.num_bins,
            device=device,
            dtype=dtype)

    def _bin_center_by_idx(self, idx: Tensor) -> Tensor:
        centers = self._bin_centers(idx.device, torch.float32)
        centers = centers.to(dtype=torch.float32 if not idx.is_floating_point()
                             else idx.dtype)
        return centers[idx]

    # ---------------------- main encode / decode ---------------------- #
    def encode(self, angle_targets: Tensor) -> Dict[str, Tensor]:
        """将 GT 角度编码为 coarse bin index + 残差角度。

        Args:
            angle_targets (Tensor): 形状 (N, 1) 或 (N,) 的角度（弧度）。
        Returns:
            dict:
                - bin_idx (Tensor): (N,) 每个样本所属的方向簇索引。
                - delta (Tensor): (N,) 对应簇中心的残差角度。
        """
        if angle_targets.dim() == 2 and angle_targets.size(-1) == 1:
            angle_targets = angle_targets[..., 0]

        # 对齐到给定的 angle_version 范围
        angle_norm = norm_angle(angle_targets, self.angle_version)

        # 粗方向簇
        bin_idx = self._angle_to_bin(angle_norm)
        center = self._bin_center_by_idx(bin_idx)

        # 簇内残差，并进行裁剪
        delta = angle_norm - center
        delta = delta.clamp(
            min=-self.max_residual,
            max=self.max_residual)

        return dict(bin_idx=bin_idx, delta=delta)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """将网络输出的编码向量解码为实际角度。

        Args:
            angle_preds (Tensor): 形状 (..., num_bins + 2) 的预测。
            keepdim (bool): 是否在最后保留一个维度。
        Returns:
            Tensor: 解码后的角度 (...,) 或 (..., 1)，弧度制。
        """
        assert angle_preds.size(-1) == self.encode_size, \
            f'Expect last dim = {self.encode_size}, got {angle_preds.size(-1)}'

        logits = angle_preds[..., :self.num_bins]
        residual_vec = angle_preds[..., self.num_bins:self.num_bins + 2]

        # (1) 粗方向：softmax 得到各簇概率，对簇中心角做期望
        prob = torch.softmax(logits, dim=-1)
        centers = self._bin_centers(
            device=angle_preds.device, dtype=angle_preds.dtype)
        theta_coarse = (prob * centers).sum(dim=-1)

        # (2) 细残差：用单位圆上的向量表示 δ
        x = residual_vec[..., 0]
        y = residual_vec[..., 1]
        norm = torch.sqrt(x * x + y * y).clamp(min=self.eps)
        x_n = x / norm
        y_n = y / norm
        delta = torch.atan2(y_n, x_n)
        # 约束在设定的残差范围内
        delta = delta.clamp(
            min=-self.max_residual,
            max=self.max_residual)

        angle = theta_coarse + delta
        angle = norm_angle(angle, self.angle_version)
        if keepdim:
            angle = angle.unsqueeze(-1)
        return angle
