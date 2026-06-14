import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl)
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.structures.bbox import norm_angle, rbbox_overlaps

INF = 1e8


@MODELS.register_module()
class COBBFCOSHead(FCOSHead):
    """COBB-style continuous OBB head adapted to an FCOS pipeline.

    The regression branch predicts 9 values per point:
    ``(dx, dy, dlog_w, dlog_h, r_ln, s0, s1, s2, s3)`` where
    ``(dx, dy, dlog_w, dlog_h, r_ln)`` parameterize the continuous OBB and
    ``s0..s3`` are style scores for the four equivalent OBB styles.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 log_scale_clamp: float = 7.0,
                 loss_style_weight: float = 0.05,
                 loss_cobb_reg_weight: float = 0.02,
                 loss_style: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.2),
                 bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='RotatedIoULoss',
                     mode='log',
                     loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs):
        self.angle_version = angle_version
        self.log_scale_clamp = float(log_scale_clamp)
        self.loss_style_weight = float(loss_style_weight)
        self.loss_cobb_reg_weight = float(loss_cobb_reg_weight)
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            **kwargs)
        self.loss_style = MODELS.build(loss_style)

    def _init_layers(self) -> None:
        FCOSHead._init_layers(self)
        self.conv_cobb = nn.Conv2d(self.feat_channels, 9, 3, padding=1)

    def forward_single(self, x: Tensor, scale, stride: int):
        cls_score, _, cls_feat, reg_feat = super(FCOSHead, self).forward_single(
            x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        cobb_pred = self.conv_cobb(reg_feat)
        return cls_score, cobb_pred, centerness

    def _flatten_stride_per_point(self, all_level_points: List[Tensor],
                                  num_imgs: int) -> Tensor:
        stride_list = []
        for level_idx, points in enumerate(all_level_points):
            stride = points.new_full((points.size(0) * num_imgs, 1),
                                     float(self.strides[level_idx]))
            stride_list.append(stride)
        return torch.cat(stride_list, dim=0)

    def _xyxyxyxy_to_rbox(self, corners: Tensor) -> Tensor:
        p1 = corners[:, 0]
        p2 = corners[:, 1]
        p3 = corners[:, 2]
        ctr = corners.mean(dim=1)
        edge1 = p2 - p1
        edge2 = p3 - p2
        width = edge1.norm(dim=-1).clamp(min=1e-6)
        height = edge2.norm(dim=-1).clamp(min=1e-6)
        angle = torch.atan2(edge1[:, 1], edge1[:, 0])
        angle = norm_angle(angle, self.angle_version)
        return torch.stack([ctr[:, 0], ctr[:, 1], width, height, angle], dim=-1)

    def _invert_rln(self, rln: Tensor) -> Tensor:
        base = torch.pow(2.0, rln - 1.0)
        rs = torch.where(rln < 0, base, 1 - base)
        return rs.clamp(min=1e-4, max=0.5)

    def _rs_to_rln(self, rs: Tensor, area_ratio: Tensor) -> Tensor:
        rs = rs.clamp(min=1e-6, max=0.5)
        return torch.where(area_ratio < 0.5, 1 + torch.log2(rs),
                           1 + torch.log2((1 - rs).clamp(min=1e-6)))

    def _cobb_reg_targets_from_rboxes(self, target_rboxes: Tensor,
                                      points: Tensor, strides: Tensor):
        if strides.ndim == 1:
            strides = strides.unsqueeze(-1)

        corners = RotatedBoxes.rbox2corner(target_rboxes)
        min_xy = corners.amin(dim=1)
        max_xy = corners.amax(dim=1)
        outer_wh = (max_xy - min_xy).clamp(min=1e-6)
        ctr = target_rboxes[:, :2]

        x_sorted, _ = corners[..., 0].sort(dim=1)
        y_sorted, _ = corners[..., 1].sort(dim=1)
        rs_h = (y_sorted[:, 1] - y_sorted[:, 0]) / outer_wh[:, 1]
        rs_v = (x_sorted[:, 1] - x_sorted[:, 0]) / outer_wh[:, 0]
        horizontal = outer_wh[:, 0] >= outer_wh[:, 1]
        rs = torch.where(horizontal, rs_h, rs_v).clamp(min=1e-6, max=0.5)

        area_ratio = (target_rboxes[:, 2] * target_rboxes[:, 3]) / (
            outer_wh[:, 0] * outer_wh[:, 1]).clamp(min=1e-6)
        rln = self._rs_to_rln(rs, area_ratio)

        dx = (ctr[:, 0] - points[:, 0]) / strides[:, 0]
        dy = (ctr[:, 1] - points[:, 1]) / strides[:, 0]
        dlog_w = torch.log((outer_wh[:, 0] / strides[:, 0]).clamp(min=1e-6))
        dlog_h = torch.log((outer_wh[:, 1] / strides[:, 0]).clamp(min=1e-6))
        reg_targets = torch.stack([dx, dy, dlog_w, dlog_h, rln], dim=-1)
        return reg_targets, outer_wh, rs

    def _build_candidate_corners(self, ctr: Tensor, outer_w: Tensor,
                                 outer_h: Tensor, rs: Tensor) -> Tensor:
        left = ctr[:, 0] - outer_w / 2
        right = ctr[:, 0] + outer_w / 2
        top = ctr[:, 1] - outer_h / 2
        bottom = ctr[:, 1] + outer_h / 2

        horizontal = outer_w >= outer_h
        term_h = (1 - 4 * (outer_h.square() / outer_w.square().clamp(
            min=1e-6)) * rs * (1 - rs)).clamp(min=1e-6)
        term_v = (1 - 4 * (outer_w.square() / outer_h.square().clamp(
            min=1e-6)) * rs * (1 - rs)).clamp(min=1e-6)
        sx_h = (1 - torch.sqrt(term_h)) * outer_w / 2
        sy_h = rs * outer_h
        sx_v = rs * outer_w
        sy_v = (1 - torch.sqrt(term_v)) * outer_h / 2

        sx = torch.where(horizontal, sx_h, sx_v)
        sy = torch.where(horizontal, sy_h, sy_v)

        l1 = torch.stack([left, top + sy], dim=-1)
        l2 = torch.stack([left, bottom - sy], dim=-1)
        t1 = torch.stack([left + sx, top], dim=-1)
        t2 = torch.stack([right - sx, top], dim=-1)
        r1 = torch.stack([right, top + sy], dim=-1)
        r2 = torch.stack([right, bottom - sy], dim=-1)
        b1 = torch.stack([left + sx, bottom], dim=-1)
        b2 = torch.stack([right - sx, bottom], dim=-1)

        return torch.stack([
            torch.stack([l1, t1, r2, b2], dim=1),
            torch.stack([t2, r1, b1, l2], dim=1),
            torch.stack([l1, t2, r2, b1], dim=1),
            torch.stack([t1, r1, b2, l2], dim=1),
        ],
                           dim=1)

    def _decode_cobb_candidates(self, ctr: Tensor, outer_w: Tensor,
                                outer_h: Tensor, rln: Tensor) -> Tensor:
        rs = self._invert_rln(rln)
        corners = self._build_candidate_corners(ctr, outer_w, outer_h, rs)
        flat_corners = corners.reshape(-1, 4, 2)
        flat_rboxes = self._xyxyxyxy_to_rbox(flat_corners)
        return flat_rboxes.reshape(-1, 4, 5)

    def _decode_cobb(self,
                     points: Tensor,
                     cobb_reg: Tensor,
                     strides: Tensor,
                     style_logits: Tensor,
                     style_inds: Optional[Tensor] = None) -> Tensor:
        if strides.ndim == 1:
            strides = strides.unsqueeze(-1)
        ctr = points + cobb_reg[:, :2] * strides
        outer_w = torch.exp(
            cobb_reg[:, 2].clamp(min=-self.log_scale_clamp,
                                 max=self.log_scale_clamp)) * strides[:, 0]
        outer_h = torch.exp(
            cobb_reg[:, 3].clamp(min=-self.log_scale_clamp,
                                 max=self.log_scale_clamp)) * strides[:, 0]
        candidates = self._decode_cobb_candidates(ctr, outer_w, outer_h,
                                                  cobb_reg[:, 4])
        if style_inds is None:
            style_inds = style_logits.sigmoid().argmax(dim=-1)
        gather_inds = style_inds.view(-1, 1, 1).expand(-1, 1, 5)
        return candidates.gather(1, gather_inds).squeeze(1)

    def _gt_rboxes_to_cobb_targets(self, target_rboxes: Tensor, points: Tensor,
                                   strides: Tensor):
        reg_targets, outer_wh, _ = self._cobb_reg_targets_from_rboxes(
            target_rboxes, points, strides)
        ctr = target_rboxes[:, :2]
        rln = reg_targets[:, 4]

        candidates = self._decode_cobb_candidates(ctr, outer_wh[:, 0],
                                                  outer_wh[:, 1], rln)
        flat_candidates = candidates.reshape(-1, 5)
        repeated_gt = target_rboxes[:, None, :].expand(-1, 4, -1).reshape(-1, 5)
        overlaps = rbbox_overlaps(flat_candidates, repeated_gt,
                                  is_aligned=True).reshape(-1, 4)
        style_targets = overlaps.clamp(min=0, max=1)
        style_inds = overlaps.argmax(dim=-1)
        return reg_targets, style_targets, style_inds

    def roundtrip_rboxes(self, target_rboxes: Tensor, points: Tensor,
                         strides: Tensor):
        """Debug helper: encode GT rboxes into COBB and decode them back."""
        reg_targets, _, _ = self._cobb_reg_targets_from_rboxes(
            target_rboxes, points, strides)
        _, _, style_inds = self._gt_rboxes_to_cobb_targets(target_rboxes,
                                                           points, strides)
        style_logits = target_rboxes.new_zeros((target_rboxes.size(0), 4))
        style_logits.scatter_(1, style_inds[:, None], 1.0)
        decoded = self._decode_cobb(points, reg_targets, strides, style_logits,
                                    style_inds=style_inds)
        return decoded, reg_targets, style_inds

    def _distance_targets_to_rbox(self, points: Tensor, bbox_targets: Tensor,
                                  angle_targets: Tensor,
                                  strides: Tensor) -> Tensor:
        if self.norm_on_bbox:
            bbox_targets = bbox_targets * strides
        distance = torch.cat([bbox_targets, angle_targets], dim=-1)
        # Reuse the existing rotated FCOS target assignment to obtain GT rboxes.
        from mmrotate.structures.bbox import distance2obb
        return distance2obb(points, distance, angle_version=self.angle_version)

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> tuple:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
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
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_targets_single(self, gt_instances: InstanceData, points: Tensor,
                            regress_ranges: Tensor,
                            num_points_per_lvl: List[int]) -> tuple:
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4)), \
                gt_bboxes.new_zeros((num_points, 1))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)

        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts,
                                                           2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle = torch.cos(gt_angle)
        sin_angle = torch.sin(gt_angle)
        rot_matrix = torch.cat(
            [cos_angle, sin_angle, -sin_angle, cos_angle],
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
        angle_targets = gt_angle[range(num_points), min_area_inds]
        return labels, bbox_targets, angle_targets

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        cobb_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(cobb_preds) == len(centernesses)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cobb_preds[0].dtype,
            device=cobb_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cobb_preds = [
            cobb_pred.permute(0, 2, 3, 1).reshape(-1, 9)
            for cobb_pred in cobb_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_cobb_preds = torch.cat(flatten_cobb_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_strides = self._flatten_stride_per_point(all_level_points,
                                                         num_imgs)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=flatten_cls_scores.device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_strides = flatten_strides[pos_inds]
            pos_cobb_preds = flatten_cobb_preds[pos_inds]

            pos_target_rboxes = self._distance_targets_to_rbox(
                pos_points, pos_bbox_targets, pos_angle_targets, pos_strides)
            reg_targets, style_targets, style_inds = self._gt_rboxes_to_cobb_targets(
                pos_target_rboxes, pos_points, pos_strides)

            pos_pred_rboxes = self._decode_cobb(
                pos_points,
                pos_cobb_preds[:, :5],
                pos_strides,
                pos_cobb_preds[:, 5:],
                style_inds=style_inds)

            bbox_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            loss_bbox = self.loss_bbox(
                pos_pred_rboxes,
                pos_target_rboxes,
                weight=pos_centerness_targets,
                avg_factor=bbox_denorm)
            if self.loss_style_weight > 0:
                loss_style = self.loss_style(
                    pos_cobb_preds[:, 5:],
                    style_targets,
                    weight=pos_centerness_targets.unsqueeze(-1).expand_as(
                        style_targets),
                    avg_factor=bbox_denorm) * self.loss_style_weight
            else:
                loss_style = pos_pred_rboxes.sum() * 0
            loss_cobb_reg = torch.nn.functional.smooth_l1_loss(
                pos_cobb_preds[:, :5],
                reg_targets,
                reduction='none',
                beta=0.15)
            loss_cobb_reg = (
                loss_cobb_reg *
                pos_centerness_targets.unsqueeze(-1)).sum() / bbox_denorm
            loss_cobb_reg = loss_cobb_reg * self.loss_cobb_reg_weight
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = flatten_cobb_preds.sum()
            loss_style = flatten_cobb_preds.sum()
            loss_cobb_reg = flatten_cobb_preds.sum()
            loss_centerness = flatten_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_style=loss_style,
            loss_cobb_reg=loss_cobb_reg,
            loss_centerness=loss_centerness)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        cobb_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True):
        assert len(cls_scores) == len(cobb_preds)

        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(len(cls_scores))]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            cobb_pred_list = select_single_mlvl(
                cobb_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(len(cls_scores))]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                cobb_pred_list=cobb_pred_list,
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
                                cobb_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        with_score_factors = score_factor_list[0] is not None
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = [] if with_score_factors else None

        for level_idx, (cls_score, cobb_pred, score_factor,
                        priors) in enumerate(
                            zip(cls_score_list, cobb_pred_list,
                                score_factor_list, mlvl_priors)):
            cobb_pred = cobb_pred.permute(1, 2, 0).reshape(-1, 9)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre, dict(cobb_pred=cobb_pred,
                                                 priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            cobb_pred = filtered_results['cobb_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            strides = priors.new_full((priors.size(0), 1),
                                      float(self.strides[level_idx]))
            decoded_bboxes = self._decode_cobb(priors, cobb_pred[:, :5],
                                               strides, cobb_pred[:, 5:])
            mlvl_bboxes.append(decoded_bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        results = InstanceData()
        if mlvl_bboxes:
            results.bboxes = RotatedBoxes(torch.cat(mlvl_bboxes, dim=0))
            results.scores = torch.cat(mlvl_scores, dim=0)
            results.labels = torch.cat(mlvl_labels, dim=0)
            if with_score_factors:
                results.score_factors = torch.cat(mlvl_score_factors, dim=0)
        else:
            empty_bboxes = cls_score_list[0].new_zeros((0, 5))
            results.bboxes = RotatedBoxes(empty_bboxes)
            results.scores = cls_score_list[0].new_zeros((0, ))
            results.labels = cls_score_list[0].new_zeros(
                (0, ), dtype=torch.long)
            if with_score_factors:
                results.score_factors = cls_score_list[0].new_zeros((0, ))

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
