import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class IntraStageSoftmaxAggregation(nn.Module):
    """Softmax history aggregation for blocks inside the same stage."""

    def __init__(self, channels, query_channels=None):
        super().__init__()
        if query_channels is None:
            query_channels = max(channels // 4, 16)

        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, query_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.key_proj = nn.Conv2d(channels, query_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, history_feats, curr_feat):
        if not history_feats:
            return curr_feat

        query = self.query_proj(curr_feat)
        scores = []
        for feat in history_feats:
            key = self.key_proj(F.adaptive_avg_pool2d(feat, 1))
            scores.append((query * key).sum(dim=1, keepdim=True))

        attn = torch.softmax(torch.stack(scores, dim=1), dim=1)

        aggregated = 0
        for idx, feat in enumerate(history_feats):
            aggregated = aggregated + attn[:, idx] * feat

        aggregated = self.out_proj(aggregated)
        return curr_feat + aggregated


class InterStageGateFusion(nn.Module):
    """Gate the previous stage feature into the current stage."""

    def __init__(self, prev_channels, curr_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(prev_channels, curr_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(curr_channels))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels * 2, curr_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_channels, curr_channels, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, prev_feat, curr_feat):
        prev_feat = self.proj(prev_feat)
        prev_feat = F.interpolate(
            prev_feat,
            size=curr_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        gate = self.gate(torch.cat([curr_feat, prev_feat], dim=1))
        return curr_feat + gate * prev_feat


@MODELS.register_module(name='hybrid_history_backbone')
class HybridHistoryBackbone(ResNet):
    """ResNet with intra-stage softmax and inter-stage gated fusion."""

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
                 use_intra_stage_softmax=True,
                 use_inter_stage_gate=True,
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

        self.use_intra_stage_softmax = use_intra_stage_softmax
        self.use_inter_stage_gate = use_inter_stage_gate

        stage_channels = [
            base_channels * (2**i) * self.block.expansion
            for i in range(num_stages)
        ]

        self.intra_stage_aggregators = nn.ModuleList()
        if self.use_intra_stage_softmax:
            for layer_name, channels in zip(self.res_layers, stage_channels):
                num_blocks = len(getattr(self, layer_name))
                self.intra_stage_aggregators.append(
                    nn.ModuleList([
                        IntraStageSoftmaxAggregation(
                            channels=channels, query_channels=query_channels)
                        for _ in range(max(num_blocks - 1, 0))
                    ]))
        else:
            for layer_name in self.res_layers:
                num_blocks = len(getattr(self, layer_name))
                self.intra_stage_aggregators.append(
                    nn.ModuleList([nn.Identity() for _ in range(max(num_blocks - 1, 0))]))

        if self.use_inter_stage_gate:
            self.inter_stage_fusions = nn.ModuleList([
                InterStageGateFusion(stage_channels[i - 1], stage_channels[i])
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
        prev_stage_feat = None
        for stage_idx, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            block_history = []
            for block_idx, block in enumerate(res_layer):
                x = block(x)
                if self.use_intra_stage_softmax and block_idx > 0:
                    x = self.intra_stage_aggregators[stage_idx][block_idx - 1](
                        block_history, x)
                block_history.append(x)

            if self.use_inter_stage_gate and prev_stage_feat is not None:
                x = self.inter_stage_fusions[stage_idx - 1](prev_stage_feat, x)

            prev_stage_feat = x
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)
