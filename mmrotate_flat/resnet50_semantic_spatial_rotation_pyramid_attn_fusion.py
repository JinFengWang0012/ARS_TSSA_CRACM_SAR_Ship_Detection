import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmrotate.registry import MODELS


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class SemanticSpatialRotationPyramidAttnFusion(nn.Module):
    """Target-aware semantic and spatial pyramid fusion."""

    def __init__(self,
                 num_levels=3,
                 feat_channels=256,
                 reduction_ratio=4,
                 alpha_init=0.1,
                 use_gate=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_levels = num_levels
        self.feat_channels = feat_channels
        self.mid_dim = max(feat_channels // reduction_ratio, 1)
        self.use_gate = use_gate

        self.query_bias = nn.Parameter(torch.zeros(num_levels, self.mid_dim))
        self.key_proj = ConvModule(
            feat_channels,
            self.mid_dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.query_proj = ConvModule(
            feat_channels,
            self.mid_dim,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.key_norm = RMSNorm(self.mid_dim)

        self.spatial_branches = nn.ModuleList([
            nn.Sequential(
                ConvModule(
                    feat_channels * 2,
                    feat_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid())
            for _ in range(num_levels)
        ])

        self.out_proj = ConvModule(
            feat_channels,
            feat_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.alpha = nn.Parameter(torch.full((num_levels,), float(alpha_init)))

        if self.use_gate:
            self.gate_layers = nn.ModuleList([
                nn.Sequential(
                    ConvModule(
                        feat_channels * 2,
                        feat_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.Conv2d(
                        feat_channels,
                        feat_channels,
                        kernel_size=1,
                        bias=True),
                    nn.Sigmoid())
                for _ in range(num_levels)
            ])

    def forward(self, laterals, target_lateral_idx, output_idx):
        target_feat = laterals[target_lateral_idx]
        current_query = self.query_proj(
            F.adaptive_avg_pool2d(target_feat, 1)).flatten(1)
        current_query = current_query + self.query_bias[output_idx].view(1, -1)

        aligned_feats = []
        key_feats = []
        spatial_masks = []
        target_size = target_feat.shape[-2:]
        for feat in laterals:
            aligned = F.interpolate(
                feat,
                size=target_size,
                mode='bilinear',
                align_corners=False)
            aligned_feats.append(aligned)

            key = self.key_proj(F.adaptive_avg_pool2d(aligned, 1)).flatten(1)
            key_feats.append(key)

            fused_input = torch.cat([aligned, target_feat], dim=1)
            spatial_mask = self.spatial_branches[output_idx](fused_input)
            spatial_masks.append(spatial_mask)

        keys = torch.stack(key_feats, dim=1)
        keys = self.key_norm(keys)
        logits = (keys * current_query.unsqueeze(1)).sum(dim=-1)
        weights = torch.softmax(logits, dim=1)

        fused = 0
        for idx, feat in enumerate(aligned_feats):
            semantic_weight = weights[:, idx].view(-1, 1, 1, 1)
            fused = fused + semantic_weight * spatial_masks[idx] * feat

        if self.use_gate:
            gate = self.gate_layers[output_idx](
                torch.cat([target_feat, fused], dim=1))
            fused = gate * fused

        fused = self.out_proj(fused)
        return target_feat + self.alpha[output_idx].view(1, 1, 1, 1) * fused


@MODELS.register_module()
class ResNet50_SemanticSpatialRotationPyramidAttnFusion(nn.Module):
    """Attention pyramid neck with semantic and spatial fusion."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 start_level=1,
                 num_outs=5,
                 reduction_ratio=4,
                 alpha_init=0.1,
                 use_gate=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert len(in_channels) >= 1
        assert 0 <= start_level < len(in_channels)
        assert num_outs >= len(in_channels) - start_level

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.start_level = start_level
        self.backbone_end_level = len(in_channels)
        self.num_outs = num_outs
        self.used_backbone_levels = self.backbone_end_level - self.start_level

        self.lateral_convs = nn.ModuleList([
            ConvModule(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in in_channels
        ])
        self.output_convs = nn.ModuleList([
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in range(self.used_backbone_levels)
        ])
        self.pyramid_fusion = SemanticSpatialRotationPyramidAttnFusion(
            num_levels=self.used_backbone_levels,
            feat_channels=out_channels,
            reduction_ratio=reduction_ratio,
            alpha_init=alpha_init,
            use_gate=use_gate,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        extra_levels = num_outs - self.used_backbone_levels
        self.extra_convs = nn.ModuleList()
        for _ in range(extra_levels):
            self.extra_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, inputs)
        ]

        pyramid_inputs = laterals[self.start_level:]
        outs = []
        for output_idx, _ in enumerate(pyramid_inputs):
            fused = self.pyramid_fusion(
                laterals,
                target_lateral_idx=output_idx + self.start_level,
                output_idx=output_idx)
            outs.append(self.output_convs[output_idx](fused))

        if self.extra_convs:
            x = outs[-1]
            for idx, extra_conv in enumerate(self.extra_convs):
                if idx > 0:
                    x = F.relu(x)
                x = extra_conv(x)
                outs.append(x)

        return tuple(outs)
