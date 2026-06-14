# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl)
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.models.utils.gaucho import (cholesky_to_gaussian,
                                          gaussian_to_rbox)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from .rotated_fcos_head import INF, RotatedFCOSHead


@MODELS.register_module()
class GauChoAP50AP75FCOSHead(RotatedFCOSHead):
    """Parallel GauCho head imported from the AP50/AP75 package."""

    def __init__(self,
                 *args,
                 regression_clamp: Tuple[float, float] = (-6.0, 4.0),
                 gamma_clamp: float = 8.0,
                 **kwargs):
        kwargs.setdefault('angle_coder', dict(type='PseudoAngleCoder'))
        kwargs.setdefault('loss_angle', None)
        kwargs.setdefault('scale_angle', False)
        super().__init__(*args, **kwargs)
        self.regression_clamp = regression_clamp
        self.gamma_clamp = float(gamma_clamp)

    def _init_layers(self):
        FCOSHead._init_layers(self)
        self.conv_gaucho = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

    def forward_single(self, x: Tensor, scale, stride: int):
        cls_score, _, cls_feat, reg_feat = super(FCOSHead,
                                                 self).forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        gaucho_pred = self.conv_gaucho(reg_feat)
        return cls_score, gaucho_pred, centerness

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], ...]:
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def _decode_gaucho(self, points: Tensor, deltas: Tensor,
                       strides: Tensor) -> Tensor:
        if strides.dim() == 1:
            strides = strides.unsqueeze(-1)

        center = points + strides * deltas[:, :2]
        log_alpha = deltas[:, 2:3].clamp(*self.regression_clamp)
        log_beta = deltas[:, 3:4].clamp(*self.regression_clamp)
        gamma_delta = deltas[:, 4:5].clamp(-self.gamma_clamp, self.gamma_clamp)

        alpha = strides * log_alpha.exp()
        beta = strides * log_beta.exp()
        gamma = strides * gamma_delta
        return torch.cat((center, alpha, beta, gamma), dim=-1)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 5))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)

        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts,
                                                           2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None]).squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        rbox_targets = gt_bboxes[range(num_points), min_area_inds]
        return labels, bbox_targets, rbox_targets

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list, rbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        rbox_targets_list = [
            rbox_targets.split(num_points, 0)
            for rbox_targets in rbox_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_rbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_rbox_targets.append(
                torch.cat(
                    [rbox_targets[i] for rbox_targets in rbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_rbox_targets

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        gaucho_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(gaucho_preds) == len(centernesses)
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
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_strides = [
            cls_score.new_full((cls_score.size(0) * cls_score.size(2) *
                                cls_score.size(3), 1), self.strides[i])
            for i, cls_score in enumerate(cls_scores)
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_gaucho_preds = torch.cat(flatten_gaucho_preds)
        flatten_centerness = torch.cat(flatten_centerness)
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
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_rbox_targets = flatten_rbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
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
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_gaucho_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        gaucho_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(gaucho_preds)
        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            gaucho_pred_list = select_single_mlvl(
                gaucho_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                gaucho_pred_list=gaucho_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                gaucho_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        with_score_factors = score_factor_list[0] is not None
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        for level_idx, (cls_score, gaucho_pred, score_factor,
                        priors) in enumerate(
                            zip(cls_score_list, gaucho_pred_list,
                                score_factor_list, mlvl_priors)):
            gaucho_pred = gaucho_pred.permute(1, 2, 0).reshape(-1, 5)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1,
                                                           self.cls_out_channels)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else \
                cls_score.softmax(-1)[:, :-1]

            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(gaucho_pred=gaucho_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            gaucho_pred = filtered_results['gaucho_pred']
            priors = filtered_results['priors']
            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            strides = priors.new_full((gaucho_pred.size(0), 1),
                                      self.strides[level_idx])
            decoded = self._decode_gaucho(priors, gaucho_pred, strides)
            xy, sigma = cholesky_to_gaussian(decoded)
            bboxes = gaussian_to_rbox(xy, sigma)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        results = InstanceData()
        results.bboxes = RotatedBoxes(torch.cat(mlvl_bboxes))
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
