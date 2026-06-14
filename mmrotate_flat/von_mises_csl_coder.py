# mmrotate/models/bbox_coders/von_mises_csl_coder.py
# 或 mmrotate/models/task_modules/coders/ 视你版本而定

import math
import torch
from torch import Tensor

from mmcv.cnn import force_fp32
from mmdet.models import BBOX_CODERS  # 旧版 mmrotate 可能是 ROTATED_BBOX_CODERS，照你本地来
from .base_angle_coder import BaseAngleCoder  # 参考现有 CSLCoder 的基类导入方式


@BBOX_CODERS.register_module()
class VonMisesCSLCoder(BaseAngleCoder):
    """CSL 改进版：用 von Mises 分布做角度编码."""

    def __init__(self,
                 angle_version='le90',
                 omega=1,
                 kappa=4.0):
        """
        Args:
            angle_version (str): 'le90', 'oc', 'le135' 等，和原来一致.
            omega (int): 每个 bin 的角度分辨率（度）.
            kappa (float): von Mises 的集中度，越大峰越窄.
        """
        super().__init__(angle_version=angle_version)
        self.omega = omega
        self.kappa = kappa

        # 根据 angle_version 决定角度范围（度）
        if angle_version == 'le90':
            self.start_deg, self.end_deg = -90.0, 90.0   # [-90, 90)
        elif angle_version == 'oc':
            self.start_deg, self.end_deg = 0.0, 180.0    # [0, 180)
        elif angle_version == 'le135':
            self.start_deg, self.end_deg = -135.0, 45.0  # [-135, 45)
        else:
            # 默认 le90
            self.start_deg, self.end_deg = -90.0, 90.0

        self.range_deg = self.end_deg - self.start_deg
        # K = 角度范围 / omega
        self.num_bins = int(round(self.range_deg / float(self.omega)))

        # 预先算好每个 bin 的中心角（弧度）
        idx = torch.arange(self.num_bins).float()
        centers_deg = self.start_deg + (idx + 0.5) * self.omega
        self.register_buffer('centers_rad',
                             centers_deg * math.pi / 180.0)

    @force_fp32(apply_to=('angle', ))
    def encode(self, angle: Tensor) -> Tensor:
        """把 GT 角度编码成 von Mises CSL 标签.

        Args:
            angle (Tensor): 形状 [..., ]，单位是弧度，范围符合 angle_version.

        Returns:
            Tensor: 形状 [..., K] 的 soft label（每条都是概率分布）.
        """
        # 展平成一维，方便操作
        angle = angle.view(-1)  # [N]
        device = angle.device

        centers = self.centers_rad.to(device)  # [K]
        K = centers.size(0)

        # angle[..., 1] (N,1) -> (N,K)
        angle_expanded = angle.unsqueeze(-1)   # [N, 1]
        # Δθ = centers - angle
        dtheta = centers.unsqueeze(0) - angle_expanded  # [N, K]

        # von Mises: p_k ∝ exp(kappa * cos(Δθ))
        kappa = self.kappa
        logits = kappa * torch.cos(dtheta)     # [N, K]
        # 为了数值稳定，减掉 max 再 exp
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        prob = torch.exp(logits)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [N, K]

        # 恢复原始形状 + K
        out_shape = angle.shape + (K, )
        prob = prob.view(out_shape).contiguous()
        return prob

    @force_fp32(apply_to=('pred', ))
    def decode(self, pred: Tensor) -> Tensor:
        """把预测的 logits / prob 解码成连续角度（弧度）.

        Args:
            pred (Tensor): 形状 [..., K]，一般是 angle head 的 logits.
        Returns:
            Tensor: 形状 [..., ] 的预测角度（弧度）.
        """
        # 先变成 prob 分布
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        N, K = pred.shape
        device = pred.device

        prob = pred.sigmoid()
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        centers = self.centers_rad.to(device)  # [K]
        cos_c = torch.cos(centers).unsqueeze(0)  # [1, K]
        sin_c = torch.sin(centers).unsqueeze(0)  # [1, K]

        mx = (prob * cos_c).sum(dim=-1)  # [N]
        my = (prob * sin_c).sum(dim=-1)  # [N]

        # 圆形均值
        angle_pred = torch.atan2(my, mx)  # [-pi, pi]

        # 然后把 angle_pred 映射回对应的 angle_version 区间（可选）
        # 这里只处理 le90 / oc / le135 的 wrap，照 mmrotate 里 angle_norm 的逻辑来
        angle_pred = self._norm_angle(angle_pred)

        return angle_pred.view(pred.shape[:-1])

    def _norm_angle(self, angle: Tensor) -> Tensor:
        """把 [-pi, pi] 的角映射回 angle_version 的范围."""
        if self.angle_version == 'le90':
            # [-90, 90) -> [-pi/2, pi/2)
            # 这里简单做一个 wrap，你可以照 mmrotate 官方 angle_norm 抄
            angle = (angle + math.pi/2) % math.pi - math.pi/2
        elif self.angle_version == 'oc':
            # [0, 180) -> [0, pi)
            angle = angle % math.pi
        elif self.angle_version == 'le135':
            # [-135, 45) -> [-3pi/4, pi/4)
            angle = (angle + 3*math.pi/4) % math.pi - 3*math.pi/4
        else:
            angle = angle
        return angle
