# mmrotate/models/necks/denoise_cbam_bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Union

from mmengine.model import BaseModule
from mmrotate.registry import MODELS


class DenoiseBlock(BaseModule):
    """轻量可学习降噪模块：残差 + 门控。

    blur = depthwise 3x3(x)
    residual = x - blur（高频）
    gate = MLP([blur, residual]) ∈ [0,1]
    out = x - gate * residual

    gate≈0 -> 基本不降噪；gate≈1 -> 接近 blur（强降噪）
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 reduction: int = 4,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert reduction >= 1
        padding = kernel_size // 2

        self.blur = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False)

        mid_channels = max(channels // reduction, 1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blur = self.blur(x)
        residual = x - blur
        gate = self.gate_net(torch.cat([blur, residual], dim=1))
        out = x - gate * residual
        return out


class CBAMBlock(BaseModule):
    """CBAM 残差块：通道注意力 + 空间注意力 + 残差."""

    def __init__(self,
                 channels: int,
                 reduction: int = 8,
                 spatial_kernel: int = 7,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = max(channels // reduction, 1)

        # channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
        )

        # spatial attention
        padding = (spatial_kernel - 1) // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca)
        x_ca = x * ca

        # Spatial attention
        avg_s = torch.mean(x_ca, dim=1, keepdim=True)
        max_s, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        out = x_ca * sa

        # residual
        out = out + x
        return out


class _DenoiseCBAMBiFPNBlock(BaseModule):
    """单个 BiFPN block：CBAM skip + 上采样/下采样降噪 + top-down + bottom-up."""

    def __init__(self,
                 num_levels: int,
                 channels: int,
                 cbam_reduction: int = 8,
                 denoise_reduction: int = 4,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        L = num_levels

        # 每层一个 CBAM（对当前尺度的“跳跃连接”做增强）
        self.cbam_blocks = nn.ModuleList([
            CBAMBlock(
                channels=channels, reduction=cbam_reduction)
            for _ in range(L)
        ])

        # top-down conv
        self.td_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1)
            for _ in range(L)
        ])

        # bottom-up conv（输出用）
        self.out_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1)
            for _ in range(L)
        ])

        # 上采样/下采样的降噪
        self.td_denoise_blocks = nn.ModuleList([
            DenoiseBlock(
                channels=channels, reduction=denoise_reduction)
            for _ in range(L - 1)
        ])
        self.bu_denoise_blocks = nn.ModuleList([
            DenoiseBlock(
                channels=channels, reduction=denoise_reduction)
            for _ in range(L - 1)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """feats: List[P2..Pn]，长度 = num_levels."""
        L = len(feats)

        # 1) skip: CBAM 残差增强
        for i in range(L):
            feats[i] = self.cbam_blocks[i](feats[i])

        # 2) top-down：上采样 + 降噪 + 融合
        td_feats: List[torch.Tensor] = [None] * L
        td_feats[-1] = self.td_convs[-1](feats[-1])  # 最高层

        for i in range(L - 2, -1, -1):
            up = F.interpolate(
                td_feats[i + 1],
                size=feats[i].shape[2:],
                mode='nearest')
            up = self.td_denoise_blocks[i](up)  # 上采样后降噪
            fusion = feats[i] + up
            td_feats[i] = self.td_convs[i](fusion)

        # 3) bottom-up：下采样 + 降噪 + 融合
        out_feats: List[torch.Tensor] = [None] * L
        out_feats[0] = self.out_convs[0](td_feats[0])

        for i in range(1, L):
            down = F.max_pool2d(out_feats[i - 1], kernel_size=2, stride=2)
            down = self.bu_denoise_blocks[i - 1](down)  # 下采样后降噪
            fusion = td_feats[i] + feats[i] + down
            out_feats[i] = self.out_convs[i](fusion)

        return out_feats


@MODELS.register_module()
class DenoiseCBAMBiFPN(BaseModule):
    """可学习降噪 + CBAM 残差增强 BiFPN（支持多轮 stack）。

    整体流程：
        C2..C5 -> 1x1 conv 对齐 -> 生成 P2..P?（含额外层）
        重复 `stack` 次：
            feats = BiFPNBlock(feats)
        返回最终 feats 作为 neck 输出。

    Args:
        in_channels (Sequence[int]): backbone 各层通道数.
        out_channels (int): FPN 输出通道数.
        start_level (int): 从哪一层 backbone 开始用（默认 C3 => 1）.
        end_level (int): 用到哪一层（-1 表示一直到最后一层）.
        num_outs (int): 最终输出的尺度数.
        add_extra_convs (bool | str): 是否用额外 conv 生成更高层.
        stack (int): BiFPN block 堆叠次数（>=1）。
        cbam_reduction (int): CBAM 通道压缩比.
        denoise_reduction (int): 降噪模块通道压缩比.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 num_outs: int = 5,
                 add_extra_convs: Union[bool, str] = True,
                 stack: int = 1,
                 cbam_reduction: int = 8,
                 denoise_reduction: int = 4,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, (list, tuple))
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.backbone_end_level - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= self.num_ins
            assert num_outs == self.backbone_end_level - start_level

        self.add_extra_convs = add_extra_convs
        self.used_backbone_levels = self.backbone_end_level - self.start_level

        # 1x1 lateral convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                self.in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            self.lateral_convs.append(l_conv)

        # extra levels
        extra_levels = self.num_outs - self.used_backbone_levels
        self.extra_convs = nn.ModuleList()
        if extra_levels > 0 and self.add_extra_convs:
            for _ in range(extra_levels):
                conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1)
                self.extra_convs.append(conv)

        assert stack >= 1
        self.stack = stack
        # 多轮 BiFPN blocks
        self.blocks = nn.ModuleList([
            _DenoiseCBAMBiFPNBlock(
                num_levels=self.num_outs,
                channels=out_channels,
                cbam_reduction=cbam_reduction,
                denoise_reduction=denoise_reduction)
            for _ in range(self.stack)
        ])

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == self.num_ins

        # 1) backbone -> 初始 FPN 特征
        feats: List[torch.Tensor] = []
        for i, l_conv in enumerate(self.lateral_convs):
            x = l_conv(inputs[i + self.start_level])
            feats.append(x)

        extra_levels = self.num_outs - self.used_backbone_levels
        if extra_levels > 0:
            if len(self.extra_convs) > 0:
                x = feats[-1]
                for conv in self.extra_convs:
                    x = conv(x)
                    feats.append(x)
            else:
                x = feats[-1]
                for _ in range(extra_levels):
                    x = F.max_pool2d(x, kernel_size=1, stride=2)
                    feats.append(x)

        # 2) 堆叠 BiFPN blocks：每一轮都是“增强 + 上下采样降噪”
        for block in self.blocks:
            feats = block(feats)

        return tuple(feats)
