import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.necks import FPN

from mmrotate.registry import MODELS


def _resize_like(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='nearest')


@MODELS.register_module()
class AugFPN(FPN):
    """A lightweight AugFPN-style approximation.

    Keeps standard FPN topology and adds a small context fusion on each level.
    """

    def __init__(self, *args, context_reduction=4, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.out_channels // context_reduction, 1)
        self.context_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.out_channels, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.out_channels, 1, bias=True),
                nn.Sigmoid())
            for _ in range(self.num_outs)
        ])

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        refined = []
        for feat, block in zip(outs, self.context_blocks):
            refined.append(feat + feat * block(feat))
        return tuple(refined)


@MODELS.register_module()
class BiFPN(nn.Module):
    """A compact BiFPN-style approximation with learnable fusion weights."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 start_level=1,
                 num_outs=5,
                 stack=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.start_level = start_level
        self.num_outs = num_outs
        self.stack = stack
        self.used_levels = len(in_channels) - start_level

        self.lateral_convs = nn.ModuleList([
            ConvModule(c, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            for c in in_channels[start_level:]
        ])
        self.td_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(self.used_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(self.used_levels - 1)
        ])
        self.out_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(self.used_levels)
        ])
        self.topdown_w = nn.Parameter(torch.ones(self.stack, self.used_levels - 1, 2))
        self.bottomup_w = nn.Parameter(torch.ones(self.stack, self.used_levels - 1, 3))

        extra_levels = num_outs - self.used_levels
        self.extra_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(extra_levels)
        ])

    def _norm_weights(self, w):
        w = F.relu(w)
        return w / (w.sum(dim=-1, keepdim=True) + 1e-4)

    def forward(self, inputs):
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs[self.start_level:])]
        for s in range(self.stack):
            td = feats[:]
            for i in range(self.used_levels - 2, -1, -1):
                w = self._norm_weights(self.topdown_w[s, i])
                fused = w[0] * td[i] + w[1] * _resize_like(td[i + 1], td[i])
                td[i] = self.td_convs[i](fused)

            bu = td[:]
            for i in range(1, self.used_levels):
                w = self._norm_weights(self.bottomup_w[s, i - 1])
                down = F.max_pool2d(bu[i - 1], kernel_size=2, stride=2)
                fused = w[0] * feats[i] + w[1] * td[i] + w[2] * down
                bu[i] = self.bu_convs[i - 1](fused)
            feats = bu

        outs = [conv(x) for conv, x in zip(self.out_convs, feats)]
        x = outs[-1]
        for conv in self.extra_convs:
            x = conv(x)
            outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class NASFPN(nn.Module):
    """A lightweight NAS-FPN-style approximation with repeated cross-level fusion."""

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 start_level=1,
                 num_outs=5,
                 stack=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.start_level = start_level
        self.num_outs = num_outs
        self.stack = stack
        self.used_levels = len(in_channels) - start_level

        self.lateral_convs = nn.ModuleList([
            ConvModule(c, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            for c in in_channels[start_level:]
        ])
        self.fuse_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(self.used_levels * stack)
        ])
        extra_levels = num_outs - self.used_levels
        self.extra_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(extra_levels)
        ])

    def forward(self, inputs):
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs[self.start_level:])]
        conv_idx = 0
        for _ in range(self.stack):
            new_feats = []
            for i, feat in enumerate(feats):
                parts = [feat]
                if i > 0:
                    parts.append(F.max_pool2d(feats[i - 1], kernel_size=2, stride=2))
                if i < len(feats) - 1:
                    parts.append(_resize_like(feats[i + 1], feat))
                fused = sum(parts) / len(parts)
                new_feats.append(self.fuse_convs[conv_idx](fused))
                conv_idx += 1
            feats = new_feats

        outs = feats[:]
        x = outs[-1]
        for conv in self.extra_convs:
            x = conv(x)
            outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class GraphFPN(nn.Module):
    """A lightweight GraphFPN-style approximation.

    Each level aggregates messages from adjacent pyramid nodes with
    learnable edge weights, then refines features by graph-style fusion.
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 start_level=1,
                 num_outs=5,
                 stack=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.start_level = start_level
        self.num_outs = num_outs
        self.stack = stack
        self.used_levels = len(in_channels) - start_level
        self.lateral_convs = nn.ModuleList([
            ConvModule(c, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            for c in in_channels[start_level:]
        ])
        self.graph_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(self.used_levels * stack)
        ])
        self.edge_weights = nn.Parameter(torch.ones(stack, self.used_levels, 3))
        extra_levels = num_outs - self.used_levels
        self.extra_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(extra_levels)
        ])

    def _norm_weights(self, w):
        w = F.relu(w)
        return w / (w.sum(dim=-1, keepdim=True) + 1e-4)

    def forward(self, inputs):
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs[self.start_level:])]
        conv_idx = 0
        for s in range(self.stack):
            new_feats = []
            for i, feat in enumerate(feats):
                weights = self._norm_weights(self.edge_weights[s, i])
                parts = [weights[0] * feat]
                if i > 0:
                    parts.append(weights[1] * F.max_pool2d(feats[i - 1], 2, 2))
                if i < len(feats) - 1:
                    parts.append(weights[2] * _resize_like(feats[i + 1], feat))
                fused = sum(parts)
                new_feats.append(self.graph_convs[conv_idx](fused))
                conv_idx += 1
            feats = new_feats
        outs = feats[:]
        x = outs[-1]
        for conv in self.extra_convs:
            x = conv(x)
            outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class RFFPNeck(FPN):
    """A lightweight approximation of a features-fused pyramid neck.

    It first performs standard FPN, then adds a second residual fusion pass
    across all pyramid levels to mimic stronger cross-scale feature reuse.
    """

    def __init__(self, *args, fusion_reduction=4, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.out_channels // fusion_reduction, 1)
        self.fuse_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.out_channels, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.out_channels, 1, bias=False))
            for _ in range(self.num_outs)
        ])
        self.refine_convs = nn.ModuleList([
            ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU'))
            for _ in range(self.num_outs)
        ])

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        fused = []
        for i, feat in enumerate(outs):
            mix = feat
            if i > 0:
                mix = mix + _resize_like(outs[i - 1], feat)
            if i < len(outs) - 1:
                mix = mix + _resize_like(outs[i + 1], feat)
            fused_feat = feat + self.fuse_proj[i](mix)
            fused.append(self.refine_convs[i](fused_feat))
        return tuple(fused)


@MODELS.register_module()
class SPAFPN(FPN):
    """A lightweight SPAFPN-style approximation.

    Uses a scarf-like multi-scale attention gate that injects pooled
    global pyramid context back into each level.
    """

    def __init__(self, *args, scarf_reduction=4, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.out_channels // scarf_reduction, 1)
        self.scarf_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.out_channels * 2, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.out_channels, 1, bias=True),
                nn.Sigmoid())
            for _ in range(self.num_outs)
        ])

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        global_context = []
        for feat in outs:
            pooled = F.adaptive_avg_pool2d(feat, 1)
            global_context.append(pooled.expand_as(feat))

        refined = []
        for feat, context, gate in zip(outs, global_context, self.scarf_gates):
            weight = gate(torch.cat([feat, context], dim=1))
            refined.append(feat + weight * context)
        return tuple(refined)
