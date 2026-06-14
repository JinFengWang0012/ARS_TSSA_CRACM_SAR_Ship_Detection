import math
from typing import Dict, Optional

import torch
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox.transforms import norm_angle
from .cf_ucr_angle_coder import CFUCRAngleCoder


@TASK_UTILS.register_module()
class AdaptiveCFUCRAngleCoder(CFUCRAngleCoder):
    """Aspect-ratio-aware CF-UCR coder for SAR-like elongated targets."""

    def __init__(self,
                 num_bins: int = 8,
                 angle_version: str = 'le90',
                 max_residual_ratio: float = 0.5,
                 square_thr: float = 1.4,
                 elongated_thr: float = 4.0,
                 min_angle_weight: float = 0.25,
                 max_angle_weight: float = 1.0,
                 min_bin_sigma: float = 0.6,
                 max_bin_sigma: float = 1.8,
                 eps: float = 1e-6) -> None:
        super().__init__(
            num_bins=num_bins,
            angle_version=angle_version,
            max_residual_ratio=max_residual_ratio,
            eps=eps)
        self.square_thr = float(square_thr)
        self.elongated_thr = float(elongated_thr)
        self.min_angle_weight = float(min_angle_weight)
        self.max_angle_weight = float(max_angle_weight)
        self.min_bin_sigma = float(min_bin_sigma)
        self.max_bin_sigma = float(max_bin_sigma)
        if self.square_thr <= 1.0:
            raise ValueError('square_thr must be greater than 1.0')
        if self.elongated_thr <= self.square_thr:
            raise ValueError('elongated_thr must be greater than square_thr')

    def _shape_confidence(self, box_wh: Optional[Tensor], dtype: torch.dtype,
                          device: torch.device) -> Tensor:
        if box_wh is None:
            return torch.ones((1, ), dtype=dtype, device=device)

        wh = box_wh.reshape(-1, 2).to(device=device, dtype=dtype)
        long_side = wh.max(dim=-1)[0].clamp_min(self.eps)
        short_side = wh.min(dim=-1)[0].clamp_min(self.eps)
        aspect_ratio = long_side / short_side

        low = math.log(self.square_thr)
        high = math.log(self.elongated_thr)
        confidence = (torch.log(aspect_ratio) - low) / max(high - low, self.eps)
        return confidence.clamp_(0.0, 1.0)

    def _soft_bin_targets(self, angle_norm: Tensor,
                          sigma_bins: Tensor) -> Tensor:
        centers = self._bin_centers(angle_norm.device, angle_norm.dtype)
        diff = angle_norm[:, None] - centers[None, :]
        diff = (diff + self.total_range * 0.5) % self.total_range \
            - self.total_range * 0.5
        sigma = (sigma_bins.clamp_min(self.eps) * self.bin_size)[:, None]
        logits = -0.5 * (diff / sigma).square()
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        prob = torch.exp(logits)
        return prob / prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def encode(self,
               angle_targets: Tensor,
               box_wh: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if angle_targets.dim() == 2 and angle_targets.size(-1) == 1:
            angle_targets = angle_targets[..., 0]

        angle_norm = norm_angle(angle_targets, self.angle_version)
        bin_idx = self._angle_to_bin(angle_norm)
        center = self._bin_center_by_idx(bin_idx)
        delta = (angle_norm - center).clamp(
            min=-self.max_residual, max=self.max_residual)

        confidence = self._shape_confidence(
            box_wh, dtype=angle_norm.dtype, device=angle_norm.device)
        if confidence.numel() == 1 and angle_norm.numel() > 1:
            confidence = confidence.expand(angle_norm.numel())

        angle_weight = self.min_angle_weight + (
            self.max_angle_weight - self.min_angle_weight) * confidence
        sigma_bins = self.max_bin_sigma - (
            self.max_bin_sigma - self.min_bin_sigma) * confidence
        bin_target = self._soft_bin_targets(angle_norm, sigma_bins)

        return dict(
            bin_idx=bin_idx,
            delta=delta,
            bin_target=bin_target,
            angle_weight=angle_weight)
