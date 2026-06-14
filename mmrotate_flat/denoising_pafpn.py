# mmrotate/models/necks/denoising_pafpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List

from mmengine.model import BaseModule
from mmdet.models.necks import PAFPN
from mmrotate.registry import MODELS


class NoiseSuppressBlock(BaseModule):
    """轻量可学习降噪模块，用在 FPN 的高分辨率层上。

    设计思路：
        - 先用 depthwise 3x3 卷积做一个低通平滑，得到 blur（低频成分）
        - 高频 residual = x - blur
        - 把 [blur, residual] 拼在一起，送入一个小的门控网络，输出与通道数相同的 gate \in [0,1]
        - 输出：blur + gate * residual
          gate 越小，高频被抑制得越狠；gate 越大，保留更多高频细节（小目标的边缘）

    这个模块完全是可学习的，不硬编码“谁是噪声”，由训练数据（比如 SAR 船）去决定。
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert reduction >= 1
        padding = kernel_size // 2

        # depthwise 低通滤波：尽量不破坏通道结构
        self.blur = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False)

        mid_channels = max(channels // reduction, 1)
        # 门控网络：输入 concat([blur, residual])，输出每个通道的 gate
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        blur = self.blur(x)
        residual = x - blur  # 高频成分

        gate = self.gate_net(torch.cat([blur, residual], dim=1))
        # 输出 = 低频 + gate * 高频
        out = blur + gate * residual
        return out


@MODELS.register_module()
class DenoisingPAFPN(PAFPN):
    """在 PAFPN 的 top-down 融合后加入降噪的 PAFPN 变体。

    和原版 PAFPN 区别：
        - 继承 PAFPN 的结构（横向连接 + 自上而下 + 自下而上聚合）
        - 在 top-down 完成后，对指定层 (denoise_levels) 进行 NoiseSuppressBlock 降噪
        - 默认在高分辨率层上做降噪（例如 level 0 和 1），主要针对小目标

    Args:
        denoise_levels (Sequence[int]): 需要降噪的 FPN 层索引（相对于 laterals 的索引）。
            laterals[0] 对应最高分辨率的 FPN 层，依次向下。
            默认 (0, 1)，即对前两层做降噪。
        denoise_reduction (int): 降噪模块中通道压缩比，越大越轻量。
        **kwargs: 其余参数直接传给 PAFPN（in_channels, out_channels, num_outs, start_level 等）
    """

    def __init__(self,
                 denoise_levels: Sequence[int] = (0, 1),
                 denoise_reduction: int = 4,
                 **kwargs) -> None:
        # 先初始化标准的 PAFPN
        super().__init__(**kwargs)

        self.denoise_levels = tuple(denoise_levels)
        self.denoise_reduction = denoise_reduction

        # PAFPN 使用的 backbone 层数范围： [start_level, backbone_end_level)
        used_levels = self.backbone_end_level - self.start_level  # = len(self.lateral_convs)

        # 每个 level 都建一个降噪模块，实际只在 denoise_levels 里启用
        self.denoise_blocks = nn.ModuleList()
        for _ in range(used_levels):
            block = NoiseSuppressBlock(
                channels=self.out_channels,
                kernel_size=3,
                reduction=denoise_reduction)
            self.denoise_blocks.append(block)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Forward function with top-down denoising."""
        assert len(inputs) == len(self.in_channels)

        # ---------- PAFPN 原版：build laterals ----------
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals[i]: FPN 对应的第 i 个横向层（从 start_level 开始）

        # ---------- PAFPN 原版：自上而下融合 ----------
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            fused = laterals[i - 1] + upsampled
            laterals[i - 1] = fused

        # ---------- 在 high-res 层上做降噪（小目标依赖的层） ----------
        for level in self.denoise_levels:
            if 0 <= level < used_backbone_levels:
                laterals[level] = self.denoise_blocks[level](laterals[level])

        # ---------- PAFPN 原版：bottom-up 聚合 + extra levels ----------
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = inter_outs[i + 1] + \
                self.downsample_convs[i](inter_outs[i])

        outs: List[torch.Tensor] = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels (和原 PAFPN 一致)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(
                        self.fpn_convs[used_backbone_levels](laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
