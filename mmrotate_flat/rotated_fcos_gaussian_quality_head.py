# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from torch import Tensor

from mmrotate.models.losses.gaussian_dist_loss import (
    gwd_loss, xy_wh_r_2_xy_sigma)
from mmrotate.registry import MODELS
from .rotated_fcos_head import RotatedFCOSHead


@MODELS.register_module()
class RotatedFCOSGaussianQualityHead(RotatedFCOSHead):
    """Rotated FCOS with Gaussian-aware quality supervision.

    The head keeps the original RotatedFCOSHead predictions. It only changes
    the centerness target into a Gaussian similarity target computed from the
    decoded predicted and assigned target rotated boxes.
    """

    def __init__(self, *args, quality_alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_alpha = quality_alpha

    def _gaussian_quality_target(self, pred_bboxes: Tensor,
                                 target_bboxes: Tensor) -> Tensor:
        pred_bboxes = pred_bboxes.detach().clone()
        target_bboxes = target_bboxes.detach().clone()
        pred_bboxes[:, 2:4] = pred_bboxes[:, 2:4].clamp(min=1e-3)
        target_bboxes[:, 2:4] = target_bboxes[:, 2:4].clamp(min=1e-3)
        distance = gwd_loss(
            xy_wh_r_2_xy_sigma(pred_bboxes),
            xy_wh_r_2_xy_sigma(target_bboxes),
            fun='none',
            tau=0.0,
            alpha=1.0,
            normalize=True,
            reduction='none')
        distance = torch.nan_to_num(distance, nan=50.0, posinf=50.0, neginf=0.0)
        distance = distance.clamp(min=0.0, max=50.0)
        quality = torch.exp(-self.quality_alpha * distance)
        return torch.nan_to_num(quality, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0, 1)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        quality_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(
            quality_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

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
        flatten_quality_preds = [
            quality_pred.permute(0, 2, 3, 1).reshape(-1)
            for quality_pred in quality_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_quality_preds = torch.cat(flatten_quality_preds)
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
        pos_quality_preds = flatten_quality_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
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

            gaussian_quality_targets = self._gaussian_quality_target(
                pos_decoded_bbox_preds, pos_decoded_target_preds)
            loss_quality = self.loss_centerness(
                pos_quality_preds, gaussian_quality_targets, avg_factor=num_pos)

            if self.loss_angle is not None:
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_quality = pos_quality_preds.sum()
            if self.loss_angle is not None:
                loss_angle = pos_angle_preds.sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_quality=loss_quality)
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_angle
        if hasattr(self.angle_coder, 'loss_angle_restrict') \
                and self.angle_coder.loss_angle_restrict is not None \
                and len(pos_inds) > 0:
            losses['loss_angle_restrict'] = \
                self.angle_coder.get_restrict_loss(pos_angle_preds)
        return losses
