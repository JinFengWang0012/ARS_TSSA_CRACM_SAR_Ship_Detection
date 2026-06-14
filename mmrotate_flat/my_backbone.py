import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class HistoryAttentionResidual(nn.Module):
    """Aggregate multiple previous stage features with attention.

    Each previous stage feature is projected into the current stage space,
    resized to the current feature resolution, and then assigned a content-
    dependent attention score. The weighted sum is added back to the current
    feature as an attention residual.
    """

    def __init__(self, prev_channels, curr_channels, attn_hidden_channels=None):
        super().__init__()
        if attn_hidden_channels is None:
            attn_hidden_channels = max(curr_channels // 4, 16)

        self.curr_pool = nn.AdaptiveAvgPool2d(1)
        self.prev_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, curr_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(curr_channels))
            for in_channels in prev_channels
        ])
        self.score_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_channels * 2, attn_hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attn_hidden_channels, 1, kernel_size=1))
            for _ in prev_channels
        ])
        self.out_gate = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, kernel_size=1, bias=True),
            nn.Sigmoid())

    def forward(self, prev_feats, curr_feat):
        if not prev_feats:
            return curr_feat

        curr_context = self.curr_pool(curr_feat)
        aligned_feats = []
        scores = []

        for prev_feat, proj, score_mlp in zip(prev_feats, self.prev_projs,
                                              self.score_mlps):
            aligned = proj(prev_feat)
            aligned = F.interpolate(
                aligned,
                size=curr_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            aligned_feats.append(aligned)

            prev_context = self.curr_pool(aligned)
            score = score_mlp(torch.cat([curr_context, prev_context], dim=1))
            scores.append(score)

        attn = torch.softmax(torch.stack(scores, dim=1), dim=1)

        residual = 0
        for idx, aligned in enumerate(aligned_feats):
            residual = residual + attn[:, idx] * aligned

        residual = self.out_gate(residual) * residual
        return curr_feat + residual


@MODELS.register_module()
class MyBackbone(ResNet):
    """ResNet backbone with multi-history attention residual aggregation.

    Compared with the previous version that only fused the immediately
    preceding stage, this version lets each stage attend to all earlier
    stages and aggregate them as an adaptive residual.
    """

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
                 attn_hidden_channels=None,
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
            self.history_attention_residuals = nn.ModuleList([
                HistoryAttentionResidual(
                    prev_channels=stage_channels[:i],
                    curr_channels=stage_channels[i],
                    attn_hidden_channels=attn_hidden_channels)
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
                x = self.history_attention_residuals[i - 1](stage_feats, x)
            stage_feats.append(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
