import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class IntraStageGatedAggregation(nn.Module):
    """Gate and aggregate multiple historical block features in the same stage."""

    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(channels // 4, 16)

        self.gate_mlps = nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.channels = channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _build_gate(self):
        return nn.Sequential(
            nn.Conv2d(self.channels * 2, self.hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, history_feats, curr_feat):
        if not history_feats:
            return curr_feat

        while len(self.gate_mlps) < len(history_feats):
            self.gate_mlps.append(self._build_gate().to(curr_feat.device))

        aggregated = 0
        curr_context = F.adaptive_avg_pool2d(curr_feat, 1)
        for feat, gate_mlp in zip(history_feats, self.gate_mlps):
            hist_context = F.adaptive_avg_pool2d(feat, 1)
            gate = gate_mlp(torch.cat([curr_context, hist_context], dim=1))
            aggregated = aggregated + gate * feat

        aggregated = self.out_proj(aggregated)
        return curr_feat + self.alpha * aggregated


class InterStageGateFusion(nn.Module):
    """Gate the previous stage feature into the current stage."""

    def __init__(self, prev_channels, curr_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(curr_channels // 4, 16)

        self.proj = nn.Sequential(
            nn.Conv2d(prev_channels, curr_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(curr_channels))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(curr_channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, curr_channels, kernel_size=1, bias=True),
            nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, prev_feat, curr_feat):
        prev_feat = self.proj(prev_feat)
        prev_feat = F.interpolate(
            prev_feat,
            size=curr_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        gate = self.gate(torch.cat([curr_feat, prev_feat], dim=1))
        return curr_feat + self.alpha * (gate * prev_feat)


@MODELS.register_module(name='gated_history_backbone')
class GatedHistoryBackbone(ResNet):
    """ResNet with gated history aggregation both intra- and inter-stage."""

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
                 use_intra_stage_gate=True,
                 use_inter_stage_gate=True,
                 gate_hidden_channels=None,
                 intra_stage_history_k=2,
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

        self.use_intra_stage_gate = use_intra_stage_gate
        self.use_inter_stage_gate = use_inter_stage_gate
        self.intra_stage_history_k = intra_stage_history_k

        stage_channels = [
            base_channels * (2**i) * self.block.expansion
            for i in range(num_stages)
        ]

        self.intra_stage_aggregators = nn.ModuleList()
        for layer_name, channels in zip(self.res_layers, stage_channels):
            num_blocks = len(getattr(self, layer_name))
            if self.use_intra_stage_gate:
                self.intra_stage_aggregators.append(
                    nn.ModuleList([
                        IntraStageGatedAggregation(
                            channels=channels, hidden_channels=gate_hidden_channels)
                        for _ in range(max(num_blocks - 1, 0))
                    ]))
            else:
                self.intra_stage_aggregators.append(
                    nn.ModuleList([nn.Identity() for _ in range(max(num_blocks - 1, 0))]))

        if self.use_inter_stage_gate:
            self.inter_stage_fusions = nn.ModuleList([
                InterStageGateFusion(
                    stage_channels[i - 1],
                    stage_channels[i],
                    hidden_channels=gate_hidden_channels)
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
                if self.use_intra_stage_gate and block_idx > 0:
                    history_feats = block_history[-self.intra_stage_history_k:]
                    x = self.intra_stage_aggregators[stage_idx][block_idx - 1](
                        history_feats, x)
                block_history.append(x)

            if self.use_inter_stage_gate and prev_stage_feat is not None:
                x = self.inter_stage_fusions[stage_idx - 1](prev_stage_feat, x)

            prev_stage_feat = x
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)