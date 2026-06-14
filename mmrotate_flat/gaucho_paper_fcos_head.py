import torch
from mmdet.models.dense_heads import FCOSHead
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from torch import Tensor

from mmrotate.registry import MODELS

from .gaucho_fcos_head import GauChoFCOSHead


@MODELS.register_module()
class GauChoPaperFCOSHead(GauChoFCOSHead):
    """A more paper-aligned GauCho head built on top of the stable baseline.

    Compared with ``GauChoFCOSHead``, this variant keeps the same FCOS-style
    assignment pipeline for compatibility, but makes two changes that are
    closer to the GauCho paper's Cholesky-centric formulation:

    1. Use the per-level FCOS ``Scale`` module on GauCho regression outputs.
    2. Add a lightweight auxiliary loss directly on the predicted Cholesky
       parameters ``(d_alpha, d_beta, d_gamma)``.
    """

    def __init__(self,
                 *args,
                 loss_cholesky_aux: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=0.15,
                     loss_weight=0.05),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cholesky_aux = (MODELS.build(loss_cholesky_aux)
                                  if loss_cholesky_aux is not None else None)

    def forward_single(self, x: Tensor, scale, stride: int):
        cls_score, _, cls_feat, reg_feat = super(FCOSHead, self).forward_single(
            x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        gaucho_pred = scale(self.conv_gaucho(reg_feat)).float()
        return cls_score, gaucho_pred, centerness

    def loss_by_feat(
        self,
        cls_scores,
        gaucho_preds,
        centernesses,
        batch_gt_instances: InstanceList,
        batch_img_metas,
        batch_gt_instances_ignore: OptInstanceList = None
    ):
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
            pos_target_gaucho = self._encode_gt_gaucho(pos_target_rboxes,
                                                       pos_points, pos_strides)
            bbox_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

            loss_bbox = self.loss_bbox(
                pos_pred_rboxes,
                pos_target_rboxes,
                weight=pos_centerness_targets,
                avg_factor=bbox_denorm)

            if self.loss_cholesky_aux is not None:
                loss_cholesky_aux = self.loss_cholesky_aux(
                    pos_gaucho_preds[:, 2:],
                    pos_target_gaucho[:, 2:],
                    weight=pos_centerness_targets.unsqueeze(-1).expand(-1, 3),
                    avg_factor=bbox_denorm)
            else:
                loss_cholesky_aux = pos_pred_rboxes.sum() * 0

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = flatten_gaucho_preds.sum()
            loss_cholesky_aux = flatten_gaucho_preds.sum()
            loss_centerness = flatten_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_cholesky_aux=loss_cholesky_aux,
            loss_centerness=loss_centerness)
