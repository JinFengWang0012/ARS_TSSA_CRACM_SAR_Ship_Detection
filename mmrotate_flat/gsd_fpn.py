# mmrotate/models/necks/gsd_fpn.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import DeformConv2dPack
from mmdet.models.necks import FPN

from mmrotate.registry import MODELS


@MODELS.register_module()
class GHSRM(nn.Module):
    """Gaussian-guided High-level Semantic Refinement Module (GHSRM).

    设计对应论文描述：
      1) Gaussian Modeling 分支：对高层语义做平滑高斯响应 G(F_in)，再残差相加得到 F_g；
      2) 1x1-AN-3x3 空洞卷积-1x1：对 F_g 做轻量空间语义建模，输出 F_s；
      3) 最后 F_out = F_in + F_s。

    这里用：
      - DWConv3x3 近似 Gaussian Modeling；
      - 1x1 -> BN+ReLU -> 3x3(dilated) -> BN+ReLU -> 1x1 的 bottleneck 结构实现高层语义精炼。
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 4,
                 dilation: int = 2) -> None:
        super().__init__()
        assert in_channels > 0
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 1)

        # 1) Gaussian Modeling 分支：DWConv3x3 + BN
        self.gauss_dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False)
        self.gauss_bn = nn.BatchNorm2d(in_channels)

        # 2) 1x1-AN-3x3(dilated)-1x1 空间语义建模
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0,
            bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian Modeling 分支
        g = self.gauss_dwconv(x)
        g = self.gauss_bn(g)
        g = self.relu(g)
        F_g = x + g  # 残差注入高斯响应

        # 1x1-AN-3x3(dilated)-1x1 语义精炼
        out = self.conv1(F_g)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 最终残差：F_out = F_in + F_s
        out = self.relu(x + out)
        return out


@MODELS.register_module()
class SDFM(nn.Module):
    """Stage-aware Deformable Fusion Module (SDFM).

    对应原 D-FCM：
      - 通道按比例拆成空间支 X_s 和语义支 X_d；
      - 语义支：3x3 DCNv2 + ReLU；
      - 空间支：1x1 Conv + BN + ReLU；
      - 语义->空间：DWConv3x3 + GAP + 1x1 Conv + Sigmoid，得到通道权重 w_c；
      - 空间->语义：Conv3x3(Cs->1) + BN + Sigmoid，得到空间权重 w_s；
      - 融合：[X~_s, X~_d] -> 1x1 Conv + BN + ReLU。
    """

    def __init__(self,
                 in_channels: int,
                 spatial_ratio: float = 0.6) -> None:
        super().__init__()
        assert 0.0 < spatial_ratio < 1.0
        assert in_channels > 1

        self.in_channels = in_channels
        self.spatial_ratio = float(spatial_ratio)

        Cs = max(1, int(round(in_channels * spatial_ratio)))
        Cd = in_channels - Cs
        assert Cd > 0, 'semantic branch channels must be > 0'

        self.Cs = Cs
        self.Cd = Cd

        # 语义支：3x3 DCNv2
        self.semantic_conv = DeformConv2dPack(
            in_channels=Cd,
            out_channels=Cd,
            kernel_size=3,
            stride=1,
            padding=1)
        self.semantic_relu = nn.ReLU(inplace=True)

        # 空间支：1x1 Conv
        self.spatial_conv = nn.Conv2d(
            Cs, Cs, kernel_size=1, stride=1, padding=0, bias=False)
        self.spatial_bn = nn.BatchNorm2d(Cs)
        self.spatial_relu = nn.ReLU(inplace=True)

        # 语义 -> 空间：通道引导
        self.dw_conv_c = nn.Conv2d(
            Cd, Cd, kernel_size=3, stride=1, padding=1, groups=Cd, bias=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Conv2d(
            Cd, Cs, kernel_size=1, stride=1, padding=0, bias=True)

        # 空间 -> 语义：空间聚合
        self.spatial_att_conv = nn.Conv2d(
            Cs, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.spatial_att_bn = nn.BatchNorm2d(1)

        # 融合输出：[X~_s, X~_d] -> 1x1 Conv + BN + ReLU
        self.fuse_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0,
            bias=False)
        self.fuse_bn = nn.BatchNorm2d(in_channels)
        self.fuse_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：先保证输入是 contiguous，避免 DCNv2 报错
        x = x.contiguous()

        B, C, H, W = x.shape
        assert C == self.in_channels
        Cs, Cd = self.Cs, self.Cd

        # 通道切片后也做 contiguous
        x_s = x[:, :Cs, :, :].contiguous()
        x_d = x[:, Cs:, :, :].contiguous()

        # 分支变换
        x_s_hat = self.spatial_conv(x_s)
        x_s_hat = self.spatial_bn(x_s_hat)
        x_s_hat = self.spatial_relu(x_s_hat)       # (B, Cs, H, W)

        # 这里之前会报 "input must be contiguous"，现在已经保证
        x_d_hat = self.semantic_relu(self.semantic_conv(x_d))  # (B, Cd, H, W)

        # 语义 -> 空间：通道引导
        u_c = self.dw_conv_c(x_d_hat)              # (B, Cd, H, W)
        v_c = self.gap(u_c)                        # (B, Cd, 1, 1)
        w_c = torch.sigmoid(self.channel_fc(v_c))  # (B, Cs, 1, 1)
        x_s_tilde = x_s_hat * w_c                  # (B, Cs, H, W)

        # 空间 -> 语义：空间聚合
        w_s = self.spatial_att_conv(x_s_tilde)     # (B, 1, H, W)
        w_s = self.spatial_att_bn(w_s)
        w_s = torch.sigmoid(w_s)                   # (B, 1, H, W)
        x_d_tilde = x_d_hat * w_s                  # (B, Cd, H, W)

        # 融合输出
        out = torch.cat([x_s_tilde, x_d_tilde], dim=1)  # (B, C, H, W)
        out = self.fuse_conv(out)
        out = self.fuse_bn(out)
        out = self.fuse_relu(out)
        return out


@MODELS.register_module()
class GSDFPN(FPN):
    """GSD-FPN: Gaussian-guided Semantic & stage-aware Deformable FPN.

    在标准 FPN 的基础上，在指定的 backbone stage 前插入可插拔模块
    （例如高层用 GHSRM，低层用 SDFM），其余逻辑沿用原始 FPN。

    Args:
        in_channels (list[int]): 各 stage 的通道数，如 [256,512,1024,2048]。
        out_channels (int): FPN 输出通道数。
        num_outs (int): 输出特征层数。
        stage_modules (list[dict]): 每个元素形如：
            dict(
                stage_idx=int,          # 对应 backbone 输出下标 (0=C2,1=C3...)
                module_cfg=dict(...),   # MODELS.build 的配置，如 type='GHSRM'
            )
        **kwargs: 其余传给 FPN 的参数（start_level/add_extra_convs/...）。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 stage_modules=None,
                 **kwargs) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        stage_modules = stage_modules if stage_modules is not None else []
        self.stage_modules_cfg = stage_modules

        modules = {}
        for cfg in stage_modules:
            stage_idx = int(cfg['stage_idx'])
            assert 0 <= stage_idx < len(in_channels)
            module_cfg = cfg['module_cfg']
            module = MODELS.build(module_cfg)
            modules[str(stage_idx)] = module

        # key 用字符串，方便 state_dict 保存
        self.stage_modules = nn.ModuleDict(modules)

    def forward(self, inputs):
        """inputs: tuple/list of backbone features, e.g. (C2, C3, C4, C5)."""
        assert len(inputs) == len(self.in_channels)

        feats = list(inputs)

        # 在指定 stage 上应用可插拔模块（GHSRM / SDFM 等）
        for k, module in self.stage_modules.items():
            idx = int(k)
            feats[idx] = module(feats[idx])

        # 再交给原始 FPN 做 top-down 融合
        return super().forward(tuple(feats))
