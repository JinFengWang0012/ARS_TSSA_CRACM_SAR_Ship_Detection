# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmdet.models.dense_heads import FCOSHead
from mmdet.utils import ConfigType, InstanceList, OptConfigType, reduce_mean
from torch import Tensor

from mmrotate.registry import MODELS
from .rotated_fcos_head import RotatedFCOSHead


@MODELS.register_module()
class DecoupledAngleAwareRotatedFCOSHead(RotatedFCOSHead):
    """Rotated FCOS with an independent angle tower and ambiguity-aware loss."""

    def __init__(self,
                 *args,
                 angle_stacked_convs: int = 2,
                 min_angle_weight: float = 0.1,
                 loss_angle: OptConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=0.1111111111111111,
                     loss_weight=0.5),
                 **kwargs):
        self.angle_stacked_convs = angle_stacked_convs
        self.min_angle_weight = min_angle_weight
        super().__init__(*args, loss_angle=loss_angle, **kwargs)

    def _init_layers(self):
        FCOSHead._init_layers(self)
        self.angle_convs = nn.ModuleList()
        for i in range(self.angle_stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.angle_stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.angle_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        if self.is_scale_angle:
            self.scale_angle = Scale(1.0)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)

        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        angle_feat = x
        for angle_layer in self.angle_convs:
            angle_feat = angle_layer(angle_feat)
        angle_pred = self.conv_angle(angle_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: InstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(
            centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        angle_dim = self.angle_coder.encode_size
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, angle_dim)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.use_hbbox_loss:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_decoded_angle_preds = self.angle_coder.decode(
                    pos_angle_preds, keepdim=True)
                pos_bbox_preds = torch.cat(
                    [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)

            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            angle_targets_encoded = self.angle_coder.encode(pos_angle_targets)
            widths = (flatten_bbox_targets[pos_inds, 0] +
                      flatten_bbox_targets[pos_inds, 2]).clamp(min=1e-6)
            heights = (flatten_bbox_targets[pos_inds, 1] +
                       flatten_bbox_targets[pos_inds, 3]).clamp(min=1e-6)
            ambiguity_weight = 1.0 - torch.minimum(widths, heights) / \
                torch.maximum(widths, heights)
            ambiguity_weight = ambiguity_weight.clamp(min=self.min_angle_weight)
            angle_weight = ambiguity_weight.unsqueeze(-1).expand_as(
                angle_targets_encoded)

            avg_factor = max(reduce_mean(ambiguity_weight.sum().detach()), 1.0)
            loss_angle = self.loss_angle(
                pos_angle_preds,
                angle_targets_encoded,
                weight=angle_weight,
                avg_factor=avg_factor)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = flatten_bbox_preds.sum() * 0
            loss_angle = flatten_angle_preds.sum() * 0
            loss_centerness = flatten_centerness.sum() * 0

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_angle=loss_angle,
            loss_centerness=loss_centerness)
        return losses
