import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class SpatialStageAttention(nn.Module):
    """Dynamic-query stage attention used before pyramid generation."""

    def __init__(self,
                 stage_channels,
                 reduction_ratio=4,
                 alpha_init=0.1,
                 history_k=2,
                 attn_start_stage=1,
                 use_gate=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.stage_channels = list(stage_channels)
        self.num_stages = len(self.stage_channels)
        self.max_channels = max(self.stage_channels)
        self.mid_dim = max(self.max_channels // reduction_ratio, 1)
        self.history_k = history_k
        self.attn_start_stage = attn_start_stage
        self.use_gate = use_gate

        self.query_bias = nn.Parameter(torch.zeros(self.num_stages, self.mid_dim))
        self.key_norm = RMSNorm(self.mid_dim)
        self.alpha = nn.Parameter(torch.full((self.num_stages,), float(alpha_init)))

        self.curr_align_layers = nn.ModuleList([
            nn.Identity() if channels == self.max_channels else ConvModule(
                channels,
                self.max_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in self.stage_channels
        ])
        self.align_layers = nn.ModuleList([
            nn.Identity() if channels == self.max_channels else ConvModule(
                channels,
                self.max_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in self.stage_channels
        ])
        self.key_layers = nn.ModuleList([
            ConvModule(
                self.max_channels,
                self.mid_dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in self.stage_channels
        ])
        self.query_layers = nn.ModuleList([
            ConvModule(
                self.max_channels,
                self.mid_dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in self.stage_channels
        ])
        self.out_layers = nn.ModuleList([
            ConvModule(
                self.max_channels,
                channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in self.stage_channels
        ])
        if self.use_gate:
            self.gate_layers = nn.ModuleList([
                nn.Sequential(
                    ConvModule(
                        self.max_channels * 2,
                        self.max_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.Conv2d(self.max_channels, self.max_channels, kernel_size=1, bias=True),
                    nn.Sigmoid())
                for _ in self.stage_channels
            ])

    def forward(self, x, stage_idx, stage_memory):
        if stage_idx < self.attn_start_stage:
            return x

        hist_start = max(0, stage_idx - self.history_k)
        mem_stage_indices = list(range(hist_start, stage_idx))
        valid_memory = stage_memory[hist_start:stage_idx]
        if not valid_memory:
            return x

        current_feat = self.curr_align_layers[stage_idx](x)
        current_query = self.query_layers[stage_idx](
            F.adaptive_avg_pool2d(current_feat, 1)).flatten(1)
        current_query = current_query + self.query_bias[stage_idx].view(1, -1)

        aligned_feats = []
        key_feats = []
        for mem_stage_idx, mem_feat in zip(mem_stage_indices, valid_memory):
            aligned = self.align_layers[mem_stage_idx](mem_feat)
            aligned = F.interpolate(
                aligned,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False)
            aligned_feats.append(aligned)

            pooled = F.adaptive_avg_pool2d(aligned, 1)
            key = self.key_layers[mem_stage_idx](pooled).flatten(1)
            key_feats.append(key)

        keys = torch.stack(key_feats, dim=1)
        keys = self.key_norm(keys)
        logits = (keys * current_query.unsqueeze(1)).sum(dim=-1)
        weights = torch.softmax(logits, dim=1)

        residual = 0
        for idx, feat in enumerate(aligned_feats):
            residual = residual + weights[:, idx].view(-1, 1, 1, 1) * feat

        if self.use_gate:
            gate = self.gate_layers[stage_idx](torch.cat([current_feat, residual], dim=1))
            residual = gate * residual

        residual = self.out_layers[stage_idx](residual)
        return x + self.alpha[stage_idx].view(1, 1, 1, 1) * residual


class PyramidAttnFusion(nn.Module):
    """Fuse C2-C5 into P3-P5 inside the backbone."""

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
                    nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True),
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

        keys = torch.stack(key_feats, dim=1)
        keys = self.key_norm(keys)
        logits = (keys * current_query.unsqueeze(1)).sum(dim=-1)
        weights = torch.softmax(logits, dim=1)

        fused = 0
        for idx, feat in enumerate(aligned_feats):
            fused = fused + weights[:, idx].view(-1, 1, 1, 1) * feat

        if self.use_gate:
            gate = self.gate_layers[output_idx](torch.cat([target_feat, fused], dim=1))
            fused = gate * fused

        fused = self.out_proj(fused)
        return target_feat + self.alpha[output_idx].view(1, 1, 1, 1) * fused


@MODELS.register_module(name='attnres_pyramid_backbone')
class AttnResPyramidBackbone(ResNet):
    """Backbone-neck integrated ResNet that directly outputs P3-P7."""

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 feat_channels=256,
                 use_stage_attnres=True,
                 use_pyramid_attn=True,
                 reduction_ratio=4,
                 alpha_init=0.1,
                 history_k=2,
                 attn_start_stage=1,
                 use_gate=True,
                 attn_act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')

        self._base_channels = base_channels
        self._num_stages = num_stages

        super().__init__(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            plugins=plugins,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            init_cfg=init_cfg)

        self.feat_channels = feat_channels
        self.use_stage_attnres = use_stage_attnres
        self.use_pyramid_attn = use_pyramid_attn

        stage_channels = self._get_stage_channels()
        if self.use_stage_attnres:
            self.stage_attn = SpatialStageAttention(
                stage_channels=stage_channels,
                reduction_ratio=reduction_ratio,
                alpha_init=alpha_init,
                history_k=history_k,
                attn_start_stage=attn_start_stage,
                use_gate=use_gate,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=attn_act_cfg)

        self.lateral_convs = nn.ModuleList([
            ConvModule(
                in_channels=channels,
                out_channels=feat_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in stage_channels
        ])
        self.output_convs = nn.ModuleList([
            ConvModule(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=attn_act_cfg)
            for _ in range(3)
        ])

        if self.use_pyramid_attn:
            self.pyramid_fusion = PyramidAttnFusion(
                num_levels=3,
                feat_channels=feat_channels,
                reduction_ratio=reduction_ratio,
                alpha_init=alpha_init,
                use_gate=use_gate,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=attn_act_cfg)

        self.p6_conv = ConvModule(
            feat_channels,
            feat_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=attn_act_cfg)
        self.p7_conv = ConvModule(
            feat_channels,
            feat_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=attn_act_cfg)

    def _get_stage_channels(self):
        return [
            self._base_channels * (2**i) * self.block.expansion
            for i in range(self._num_stages)
        ]

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        stage_feats = []
        stage_memory = []
        for stage_idx, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if self.use_stage_attnres:
                x = self.stage_attn(x, stage_idx, stage_memory)
            stage_memory.append(x)
            stage_feats.append(x)

        laterals = [
            lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, stage_feats)
        ]

        pyramid_inputs = laterals[1:]
        pyramid_outs = []
        for output_idx, target_feat in enumerate(pyramid_inputs):
            if self.use_pyramid_attn:
                fused = self.pyramid_fusion(
                    laterals, target_lateral_idx=output_idx + 1, output_idx=output_idx)
            else:
                fused = target_feat
            pyramid_outs.append(self.output_convs[output_idx](fused))

        p3, p4, p5 = pyramid_outs
        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(F.relu(p6))
        return (p3, p4, p5, p6, p7)
