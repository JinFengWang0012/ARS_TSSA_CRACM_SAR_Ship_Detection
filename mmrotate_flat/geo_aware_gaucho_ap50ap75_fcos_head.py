# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from torch import Tensor

from mmrotate.registry import MODELS
from .gaucho_ap50ap75_fcos_head import GauChoAP50AP75FCOSHead


@MODELS.register_module()
class GeometryAwareGauChoAP50AP75FCOSHead(GauChoAP50AP75FCOSHead):
    """Geometry-aware AP50/AP75 GauCho head imported as a parallel variant."""

    def __init__(self,
                 *args,
                 quality_hidden_channels: int = 64,
                 quality_aspect_thr: float = 1.6,
                 quality_aspect_scale: float = 3.0,
                 min_quality_weight: float = 0.4,
                 detach_geom: bool = True,
                 **kwargs):
        self.quality_hidden_channels = int(quality_hidden_channels)
        self.quality_aspect_thr = float(quality_aspect_thr)
        self.quality_aspect_scale = float(quality_aspect_scale)
        self.min_quality_weight = float(min_quality_weight)
        self.detach_geom = bool(detach_geom)
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        self.geom_proj = ConvModule(
            5,
            self.quality_hidden_channels,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))
        self.quality_conv = nn.Sequential(
            ConvModule(
                self.feat_channels + self.quality_hidden_channels,
                self.feat_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU')),
            nn.Conv2d(self.feat_channels, 1, kernel_size=3, padding=1))

    def _geometry_confidence_target(self, rbox_targets: Tensor) -> Tensor:
        wh = rbox_targets[:, 2:4]
        long_side = wh.max(dim=-1)[0].clamp_min(1e-6)
        short_side = wh.min(dim=-1)[0].clamp_min(1e-6)
        aspect_ratio = long_side / short_side
        geometry_conf = torch.sigmoid((aspect_ratio - self.quality_aspect_thr)
                                      * self.quality_aspect_scale)
        return self.min_quality_weight + \
            (1.0 - self.min_quality_weight) * geometry_conf

    def forward_single(self, x: Tensor, scale, stride: int):
        cls_score, gaucho_pred, centerness = super().forward_single(
            x, scale, stride)
        geom_input = gaucho_pred.detach() if self.detach_geom else gaucho_pred
        geom_feat = self.geom_proj(geom_input)
        quality_logit = self.quality_conv(torch.cat([x, geom_feat], dim=1))
        quality_pred = centerness + quality_logit
        return cls_score, gaucho_pred, quality_pred

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        gaucho_preds: List[Tensor],
        quality_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(gaucho_preds) == len(quality_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=gaucho_preds[0].dtype,
            device=gaucho_preds[0].device)
        labels, bbox_targets, rbox_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_gaucho_preds = [
            gaucho_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            for gaucho_pred in gaucho_preds
        ]
        flatten_quality_preds = [
            quality_pred.permute(0, 2, 3, 1).reshape(-1)
            for quality_pred in quality_preds
        ]
        flatten_strides = [
            cls_score.new_full((cls_score.size(0) * cls_score.size(2) *
                                cls_score.size(3), 1), self.strides[i])
            for i, cls_score in enumerate(cls_scores)
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_gaucho_preds = torch.cat(flatten_gaucho_preds)
        flatten_quality_preds = torch.cat(flatten_quality_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_rbox_targets = torch.cat(rbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_strides = torch.cat(flatten_strides)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=gaucho_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_gaucho_preds = flatten_gaucho_preds[pos_inds]
        pos_quality_preds = flatten_quality_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_rbox_targets = flatten_rbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        pos_geometry_targets = self._geometry_confidence_target(pos_rbox_targets)
        pos_quality_targets = pos_centerness_targets * pos_geometry_targets
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_strides = flatten_strides[pos_inds]
            pos_decoded_gaucho = self._decode_gaucho(pos_points, pos_gaucho_preds,
                                                     pos_strides)
            loss_bbox = self.loss_bbox(
                pos_decoded_gaucho,
                pos_rbox_targets,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_quality = self.loss_centerness(
                pos_quality_preds, pos_quality_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_gaucho_preds.sum()
            loss_quality = flatten_quality_preds.sum() * 0

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_quality=loss_quality)
