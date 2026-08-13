# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox.transforms import norm_angle


@TASK_UTILS.register_module(name='CRACMCoder')
class CRACMCoder(BaseBBoxCoder):
    """Consistency-Regularized Angle Modeling coder.

    CR-ACM represents an angle with dual-frequency unit-circle codes:
    [cos(w*a), sin(w*a), cos(2*w*a), sin(2*w*a)]. With the paper setting
    base_omega=2 and dual_freq=True, the encoded angle dimension is 4.
    """

    def __init__(self,
                 angle_version: str = 'le90',
                 base_omega: int = 2,
                 dual_freq: bool = True,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.angle_version = angle_version
        self.base_omega = int(base_omega)
        self.dual_freq = bool(dual_freq)
        self.eps = float(eps)
        self.encode_size = 4 if self.dual_freq else 2

    def encode(self, angle_targets: Tensor) -> Tensor:
        if angle_targets.dim() == 2 and angle_targets.size(-1) == 1:
            angle_targets = angle_targets[..., 0]

        angle_targets = norm_angle(angle_targets, self.angle_version)
        omega = float(self.base_omega)
        code = [
            torch.cos(omega * angle_targets),
            torch.sin(omega * angle_targets),
        ]
        if self.dual_freq:
            code.extend([
                torch.cos(2.0 * omega * angle_targets),
                torch.sin(2.0 * omega * angle_targets),
            ])
        return torch.stack(code, dim=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        assert angle_preds.size(-1) == self.encode_size, \
            f'Expect last dim = {self.encode_size}, got {angle_preds.size(-1)}'

        cos_w = angle_preds[..., 0]
        sin_w = angle_preds[..., 1]
        norm_w = torch.sqrt(cos_w.square() + sin_w.square()).clamp_min(self.eps)
        angle = torch.atan2(sin_w / norm_w, cos_w / norm_w) / float(self.base_omega)

        # The second frequency is regularized by CRACMConsistencyLoss during
        # training; decoding uses the fundamental frequency for continuity.
        angle = norm_angle(angle, self.angle_version)
        if keepdim:
            angle = angle.unsqueeze(-1)
        return angle