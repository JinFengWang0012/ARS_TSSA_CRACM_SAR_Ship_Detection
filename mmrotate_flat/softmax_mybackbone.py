import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class StageSoftmaxHistoryAggregation(nn.Module):
    """Stage-level softmax aggregation over all historical stage features."""

    def __init__(self, prev_channels, curr_channels, query_channels=None):
        super().__init__()
        if query_channels is None:
            query_channels = max(curr_channels // 4, 16)

        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels, query_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.key_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, curr_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(curr_channels))
            for in_channels in prev_channels
        ])
        self.key_score_projs = nn.ModuleList([
            nn.Conv2d(curr_channels, query_channels, kernel_size=1, bias=False)
            for _ in prev_channels
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(curr_channels))

    def forward(self, prev_feats, curr_feat):
        if not prev_feats:
            return curr_feat

        query = self.query_proj(curr_feat)
        aligned_feats = []
        scores = []

        for prev_feat, key_proj, key_score_proj in zip(prev_feats,
                                                       self.key_projs,
                                                       self.key_score_projs):
            aligned = key_proj(prev_feat)
            aligned = F.interpolate(
                aligned,
                size=curr_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            aligned_feats.append(aligned)

            key = F.adaptive_avg_pool2d(aligned, 1)
            key = key_score_proj(key)
            score = (query * key).sum(dim=1, keepdim=True)
            scores.append(score)

        attn = torch.softmax(torch.stack(scores, dim=1), dim=1)

        aggregated = 0
        for idx, aligned in enumerate(aligned_feats):
            aggregated = aggregated + attn[:, idx] * aligned

        aggregated = self.out_proj(aggregated)
        return curr_feat + aggregated


@MODELS.register_module(name='softmax_mybackbone')
class SoftmaxMyBackbone(ResNet):
    """ResNet backbone with stage-level softmax history aggregation."""

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
                 use_stage_attention_residual=True,
                 query_channels=None,
                 init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')

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
        self.use_stage_attention_residual = use_stage_attention_residual

        if self.use_stage_attention_residual:
            stage_channels = [
                base_channels * (2**i) * self.block.expansion
                for i in range(num_stages)
            ]
            self.history_aggregators = nn.ModuleList([
                StageSoftmaxHistoryAggregation(
                    prev_channels=stage_channels[:i],
                    curr_channels=stage_channels[i],
                    query_channels=query_channels)
                for i in range(1, num_stages)
            ])

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        stage_feats = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.use_stage_attention_residual and i > 0:
                x = self.history_aggregators[i - 1](stage_feats, x)
            stage_feats.append(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
