import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.backbones import ResNet

from mmrotate.registry import MODELS


class RMSNorm(nn.Module):
    """RMSNorm used to normalize historical stage keys."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class SpatialStageAttention(nn.Module):
    """Stage-level attention module separated from the backbone body."""

    def __init__(self,
                 stage_channels,
                 attn_embed_channels=None,
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
        self.embed_channels = (
            max(self.stage_channels)
            if attn_embed_channels is None else int(attn_embed_channels))
        self.mid_dim = max(self.embed_channels // reduction_ratio, 1)
        self.history_k = history_k
        self.attn_start_stage = attn_start_stage
        self.use_gate = use_gate

        self.query_bias = nn.Parameter(
            torch.zeros(self.num_stages, self.mid_dim))
        self.key_norm = RMSNorm(self.mid_dim)
        self.alpha = nn.Parameter(
            torch.full((self.num_stages,), float(alpha_init)))

        self.curr_align_layers = nn.ModuleList([
            nn.Identity() if channels == self.embed_channels else ConvModule(
                channels,
                self.embed_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in self.stage_channels
        ])
        self.align_layers = nn.ModuleList([
            nn.Identity() if channels == self.embed_channels else ConvModule(
                channels,
                self.embed_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            for channels in self.stage_channels
        ])
        self.key_layers = nn.ModuleList([
            ConvModule(
                self.embed_channels,
                self.mid_dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in self.stage_channels
        ])
        self.query_layers = nn.ModuleList([
            ConvModule(
                self.embed_channels,
                self.mid_dim,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            for _ in self.stage_channels
        ])
        self.out_layers = nn.ModuleList([
            ConvModule(
                self.embed_channels,
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
                        self.embed_channels * 2,
                        self.embed_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    nn.Conv2d(self.embed_channels, self.embed_channels, kernel_size=1, bias=True),
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


@MODELS.register_module(name='attnres_stage_backbone')
class AttnResStageBackbone(ResNet):
    """ResNet backbone with modular stage-level attention residuals."""

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
                 use_attnres=True,
                 attn_embed_channels=None,
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
        self._attn_conv_cfg = conv_cfg
        self._attn_norm_cfg = norm_cfg
        self._attn_act_cfg = attn_act_cfg
        self._attn_embed_channels = attn_embed_channels
        self._reduction_ratio = reduction_ratio
        self._alpha_init = alpha_init
        self._history_k = history_k
        self._attn_start_stage = attn_start_stage
        self._use_gate = use_gate

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
        self.use_attnres = use_attnres

        if self.use_attnres:
            self.spatial_attn = SpatialStageAttention(
                stage_channels=self._get_stage_channels(),
                attn_embed_channels=self._attn_embed_channels,
                reduction_ratio=self._reduction_ratio,
                alpha_init=self._alpha_init,
                history_k=self._history_k,
                attn_start_stage=self._attn_start_stage,
                use_gate=self._use_gate,
                conv_cfg=self._attn_conv_cfg,
                norm_cfg=self._attn_norm_cfg,
                act_cfg=self._attn_act_cfg)

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

        outs = []
        stage_memory = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.use_attnres:
                x = self.spatial_attn(x, i, stage_memory)
            stage_memory.append(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
