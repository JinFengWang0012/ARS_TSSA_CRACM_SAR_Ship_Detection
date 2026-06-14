import torch
import torch.nn as nn
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ECABlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class _AttentionResNetBase(ResNet):
    stage_channels = [256, 512, 1024, 2048]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_blocks = nn.ModuleList(
            [self.build_attention_block(self.stage_channels[idx]) for idx in self.out_indices])

    def build_attention_block(self, channels):
        raise NotImplementedError

    def forward(self, x):
        outs = super().forward(x)
        return tuple(
            block(feat) for block, feat in zip(self.attn_blocks, outs))


@MODELS.register_module()
class SEResNetBackbone(_AttentionResNetBase):
    def __init__(self, *args, se_reduction=16, **kwargs):
        self.se_reduction = se_reduction
        super().__init__(*args, **kwargs)

    def build_attention_block(self, channels):
        return SEBlock(channels, reduction=self.se_reduction)


@MODELS.register_module()
class ECABackbone(_AttentionResNetBase):
    def __init__(self, *args, eca_kernel_size=3, **kwargs):
        self.eca_kernel_size = eca_kernel_size
        super().__init__(*args, **kwargs)

    def build_attention_block(self, channels):
        return ECABlock(channels, kernel_size=self.eca_kernel_size)


@MODELS.register_module()
class CBAMResNetBackbone(_AttentionResNetBase):
    def __init__(self, *args, cbam_reduction=16, cbam_spatial_kernel=7, **kwargs):
        self.cbam_reduction = cbam_reduction
        self.cbam_spatial_kernel = cbam_spatial_kernel
        super().__init__(*args, **kwargs)

    def build_attention_block(self, channels):
        return CBAMBlock(
            channels,
            reduction=self.cbam_reduction,
            spatial_kernel=self.cbam_spatial_kernel)
