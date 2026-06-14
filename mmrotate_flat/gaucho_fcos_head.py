import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl)
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.structures.bbox import distance2obb, norm_angle

INF = 1e8


@MODELS.register_module()
class GauChoFCOSHead(FCOSHead):
    r"""Anchor-free GauCho head for oriented detection.

    This head follows the anchor-free parameterization described in
    "Gaussian Distributions with Cholesky Decomposition for Oriented Object
    Detection" (CVPR 2025). For each point on an FPN level with stride ``t``,
    it predicts:

    - ``dx, dy``: normalized center offsets
    - ``d_alpha, d_beta, d_gamma``: normalized Cholesky parameters

    The decoded Gaussian is

    - ``x = p_x + t * dx``
    - ``y = p_y + t * dy``
    - ``alpha = t * exp(d_alpha)``
    - ``beta = t * exp(d_beta)``
    - ``gamma = t * d_gamma``

    with covariance

    - ``Sigma = [[alpha^2, alpha * gamma], [alpha * gamma, beta^2 + gamma^2]]``

    The covariance is converted back to an oriented box for inference and for
    Gaussian-based regression losses such as ``GDLoss``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 gaussian_scale: float = 0.25,
                 log_scale_clamp: float = 7.0,
                 bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GDLoss',
                     loss_type='kld',
                     loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs):
        self.angle_version = angle_version
        self.gaussian_scale = float(gaussian_scale)
        self.log_scale_clamp = float(log_scale_clamp)
        if self.gaussian_scale <= 0:
            raise ValueError('gaussian_scale must be positive.')

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize the FCOS stem plus the GauCho regression branch."""
        FCOSHead._init_layers(self)
        self.conv_gaucho = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

    def forward_single(self, x: Tensor, scale, stride: int):
        """Forward features of a single scale level."""
        cls_score, _, cls_feat, reg_feat = super(FCOSHead, self).forward_single(
            x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        gaucho_pred = self.conv_gaucho(reg_feat)
        return cls_score, gaucho_pred, centerness

    def _flatten_stride_per_point(self, all_level_points: List[Tensor],
                                  num_imgs: int) -> Tensor:
        stride_list = []
        for level_idx, points in enumerate(all_level_points):
            stride = points.new_full((points.size(0) * num_imgs, 1),
                                     float(self.strides[level_idx]))
            stride_list.append(stride)
        return torch.cat(stride_list, dim=0)

    def _regularized_covariance_to_rbox(self, ctr: Tensor, alpha: Tensor,
                                        beta: Tensor,
                                        gamma: Tensor) -> Tensor:
        sigma_xx = alpha.square()
        sigma_xy = alpha * gamma
        sigma_yy = beta.square() + gamma.square()

        trace = sigma_xx + sigma_yy
        delta = torch.sqrt((sigma_xx - sigma_yy).square() +
                           4 * sigma_xy.square() + 1e-9)
        eig_major = ((trace + delta) * 0.5).clamp(min=1e-9)
        eig_minor = ((trace - delta) * 0.5).clamp(min=1e-9)

        w = torch.sqrt(eig_major / self.gaussian_scale)
        h = torch.sqrt(eig_minor / self.gaussian_scale)
        angle = 0.5 * torch.atan2(2 * sigma_xy, sigma_xx - sigma_yy)
        angle = norm_angle(angle, self.angle_version)
        return torch.stack([ctr[:, 0], ctr[:, 1], w, h, angle], dim=-1)

    def _decode_gaucho(self, points: Tensor, gaucho_pred: Tensor,
                       strides: Tensor) -> Tensor:
        if strides.ndim == 1:
            strides = strides.unsqueeze(-1)

        ctr = points + gaucho_pred[:, :2] * strides
        alpha = torch.exp(
            gaucho_pred[:, 2].clamp(min=-self.log_scale_clamp,
                                    max=self.log_scale_clamp)) * strides[:, 0]
        beta = torch.exp(
            gaucho_pred[:, 3].clamp(min=-self.log_scale_clamp,
                                    max=self.log_scale_clamp)) * strides[:, 0]
        gamma = gaucho_pred[:, 4] * strides[:, 0]
        return self._regularized_covariance_to_rbox(ctr, alpha, beta, gamma)

    def _encode_gt_gaucho(self, target_rboxes: Tensor, points: Tensor,
                          strides: Tensor) -> Tensor:
        if strides.ndim == 1:
            strides = strides.unsqueeze(-1)

        x, y, w, h, angle = target_rboxes.unbind(dim=-1)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        lambda_w = self.gaussian_scale * w.square()
        lambda_h = self.gaussian_scale * h.square()

        sigma_xx = lambda_w * cos_angle.square() + lambda_h * sin_angle.square()
        sigma_xy = (lambda_w - lambda_h) * sin_angle * cos_angle
        sigma_yy = lambda_w * sin_angle.square() + lambda_h * cos_angle.square()

        alpha = torch.sqrt(sigma_xx.clamp(min=1e-9))
        gamma = sigma_xy / alpha.clamp(min=1e-9)
        beta = torch.sqrt((sigma_yy - gamma.square()).clamp(min=1e-9))

        dx = (x - points[:, 0]) / strides[:, 0]
        dy = (y - points[:, 1]) / strides[:, 0]
        d_alpha = torch.log((alpha / strides[:, 0]).clamp(min=1e-9))
        d_beta = torch.log((beta / strides[:, 0]).clamp(min=1e-9))
        d_gamma = gamma / strides[:, 0]
        return torch.stack([dx, dy, d_alpha, d_beta, d_gamma], dim=-1)

    def _distance_targets_to_rbox(self, points: Tensor, bbox_targets: Tensor,
                                  angle_targets: Tensor,
                                  strides: Tensor) -> Tensor:
        if self.norm_on_bbox:
            bbox_targets = bbox_targets * strides
        distance = torch.cat([bbox_targets, angle_targets], dim=-1)
        return distance2obb(points, distance, angle_version=self.angle_version)

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> tuple:
        """Compute labels, distance targets and angle targets."""
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
        """Compute rotated FCOS-style targets for a single image."""
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
        gaucho_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate losses from GauCho predictions."""
        del batch_img_metas
        del batch_gt_instances_ignore
        assert len(cls_scores) == len(gaucho_preds) == len(centernesses)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=gaucho_preds[0].dtype,
            device=gaucho_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(
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

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_gaucho_preds = torch.cat(flatten_gaucho_preds)
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
            pos_gaucho_preds = flatten_gaucho_preds[pos_inds]

            pos_target_rboxes = self._distance_targets_to_rbox(
                pos_points, pos_bbox_targets, pos_angle_targets, pos_strides)
            pos_pred_rboxes = self._decode_gaucho(pos_points, pos_gaucho_preds,
                                                  pos_strides)
            bbox_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

            loss_bbox = self.loss_bbox(
                pos_pred_rboxes,
                pos_target_rboxes,
                weight=pos_centerness_targets,
                avg_factor=bbox_denorm)

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = flatten_gaucho_preds.sum()
            loss_centerness = flatten_centerness.sum()

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
        """Transform network outputs into rotated detection results."""
        assert len(cls_scores) == len(gaucho_preds)

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
            gaucho_pred_list = select_single_mlvl(
                gaucho_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(len(cls_scores))]

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
        """Transform a single image's features into bbox results."""
        with_score_factors = score_factor_list[0] is not None
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = [] if with_score_factors else None

        for level_idx, (cls_score, gaucho_pred, score_factor,
                        priors) in enumerate(
                            zip(cls_score_list, gaucho_pred_list,
                                score_factor_list, mlvl_priors)):
            gaucho_pred = gaucho_pred.permute(1, 2, 0).reshape(-1, 5)
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
                scores, score_thr, nms_pre,
                dict(gaucho_pred=gaucho_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            gaucho_pred = filtered_results['gaucho_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            strides = priors.new_full((priors.size(0), 1),
                                      float(self.strides[level_idx]))
            decoded_bboxes = self._decode_gaucho(priors, gaucho_pred, strides)

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
