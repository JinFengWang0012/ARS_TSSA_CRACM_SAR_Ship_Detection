# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl)
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from .rotated_fcos_head import RotatedFCOSHead


@MODELS.register_module()
class RotatedFCOSPQAHead(RotatedFCOSHead):
    """Rotated FCOS with PQA quality branch and independent angle-quality."""

    def __init__(self,
                 *args,
                 quality_alpha: float = 2.0,
                 angle_quality_beta: float = 1.5,
                 loss_angle_quality: Optional[dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_alpha = quality_alpha
        self.angle_quality_beta = angle_quality_beta
        if loss_angle_quality is None:
            loss_angle_quality = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=0.2)
        self.loss_angle_quality = MODELS.build(loss_angle_quality)

    def _init_layers(self):
        super()._init_layers()
        self.conv_angle_quality = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def forward_single(self, x: Tensor, scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor,
                                             Tensor]:
        cls_score, bbox_pred, cls_feat, reg_feat = super(
            FCOSHead, self).forward_single(x)
        if self.centerness_on_reg:
            quality_pred = self.conv_centerness(reg_feat)
        else:
            quality_pred = self.conv_centerness(cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.is_scale_angle:
            angle_pred = self.scale_angle(angle_pred).float()
        angle_quality_pred = self.conv_angle_quality(reg_feat)
        return cls_score, bbox_pred, angle_pred, quality_pred, angle_quality_pred

    def _quality_targets_from_bbox_targets(
            self, lvl_labels: Tensor,
            lvl_bbox_targets: Tensor) -> Tuple[Tensor, Tensor]:
        width = (lvl_bbox_targets[:, 0] + lvl_bbox_targets[:, 2]).clamp(min=1e-6)
        height = (lvl_bbox_targets[:, 1] + lvl_bbox_targets[:, 3]).clamp(
            min=1e-6)
        offset_x = 0.5 * (lvl_bbox_targets[:, 0] - lvl_bbox_targets[:, 2])
        offset_y = 0.5 * (lvl_bbox_targets[:, 1] - lvl_bbox_targets[:, 3])
        norm_dist2 = (2 * offset_x / width).square() + (
            2 * offset_y / height).square()
        quality_targets = torch.exp(-self.quality_alpha * norm_dist2)
        quality_targets = quality_targets.clamp_(0, 1)

        log_ratio = torch.abs(torch.log(width / height))
        angle_quality_targets = 1 - torch.exp(-self.angle_quality_beta * log_ratio)
        angle_quality_targets = angle_quality_targets.clamp_(0, 1)

        bg_mask = lvl_labels == self.num_classes
        quality_targets[bg_mask] = 0
        angle_quality_targets[bg_mask] = 0
        return quality_targets, angle_quality_targets

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor],
               List[Tensor]]:
        labels, bbox_targets, angle_targets = super().get_targets(
            points, batch_gt_instances)
        quality_targets = []
        angle_quality_targets = []
        for lvl_labels, lvl_bbox_targets in zip(labels, bbox_targets):
            lvl_quality, lvl_angle_quality = self._quality_targets_from_bbox_targets(
                lvl_labels, lvl_bbox_targets)
            quality_targets.append(lvl_quality)
            angle_quality_targets.append(lvl_angle_quality)
        return (labels, bbox_targets, angle_targets, quality_targets,
                angle_quality_targets)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        quality_preds: List[Tensor],
        angle_quality_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(
            quality_preds) == len(angle_quality_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        (labels, bbox_targets, angle_targets, quality_targets,
         angle_quality_targets) = self.get_targets(all_level_points,
                                                   batch_gt_instances)

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
        flatten_angle_quality_preds = [
            angle_quality_pred.permute(0, 2, 3, 1).reshape(-1)
            for angle_quality_pred in angle_quality_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_quality_preds = torch.cat(flatten_quality_preds)
        flatten_angle_quality_preds = torch.cat(flatten_angle_quality_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_quality_targets = torch.cat(quality_targets)
        flatten_angle_quality_targets = torch.cat(angle_quality_targets)
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

        loss_quality = self.loss_centerness(
            flatten_quality_preds, flatten_quality_targets, avg_factor=num_pos)
        loss_angle_quality = self.loss_angle_quality(
            flatten_angle_quality_preds,
            flatten_angle_quality_targets,
            avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
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
            if self.loss_angle is not None:
                pos_angle_targets = self.angle_coder.encode(pos_angle_targets)
                loss_angle = self.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            if self.loss_angle is not None:
                loss_angle = pos_angle_preds.sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_quality=loss_quality,
            loss_angle_quality=loss_angle_quality)
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_angle
        if hasattr(self.angle_coder, 'loss_angle_restrict') \
                and self.angle_coder.loss_angle_restrict is not None \
                and len(pos_inds) > 0:
            losses['loss_angle_restrict'] = \
                self.angle_coder.get_restrict_loss(pos_angle_preds)
        return losses

    def predict_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            quality_preds: List[Tensor],
            angle_quality_preds: List[Tensor],
            batch_img_metas: Optional[List[dict]] = None,
            cfg: Optional[ConfigDict] = None,
            rescale: bool = False,
            with_nms: bool = True):
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            quality_pred_list = select_single_mlvl(
                quality_preds, img_id, detach=True)
            angle_quality_pred_list = select_single_mlvl(
                angle_quality_preds, img_id, detach=True)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                quality_pred_list=quality_pred_list,
                angle_quality_pred_list=angle_quality_pred_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                quality_pred_list: List[Tensor],
                                angle_quality_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = []
        mlvl_angle_score_factors = []
        for cls_score, bbox_pred, angle_pred, quality_pred, angle_quality_pred, \
                priors in zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              quality_pred_list, angle_quality_pred_list,
                              mlvl_priors):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            quality_pred = quality_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            angle_quality_pred = angle_quality_pred.permute(1, 2,
                                                            0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else \
                cls_score.softmax(-1)[:, :-1]
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    angle_pred=angle_pred,
                    priors=priors,
                    quality_pred=quality_pred,
                    angle_quality_pred=angle_quality_pred))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']
            quality_pred = filtered_results['quality_pred']
            angle_quality_pred = filtered_results['angle_quality_pred']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(quality_pred)
            mlvl_angle_score_factors.append(angle_quality_pred)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        results.score_factors = torch.cat(mlvl_score_factors)
        results.angle_score_factors = torch.cat(mlvl_angle_score_factors)
        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        if hasattr(results, 'angle_score_factors'):
            angle_score_factors = results.angle_score_factors
            del results.angle_score_factors
            results.score_factors = results.score_factors * angle_score_factors
        return super()._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
