# mmrotate/models/utils/sar_fpn_denoiser.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import roi_align
from mmrotate.registry import MODELS


def obb_to_xyxy(rbboxes: torch.Tensor) -> torch.Tensor:
    """旋转框 (cx, cy, w, h, theta) -> 外接水平框 (x1, y1, x2, y2).

    这里假设 rbboxes 的最后一维是 [cx, cy, w, h, theta]。
    theta 单位是弧度/角度无所谓，我们只用它做 cos/sin。
    """
    if rbboxes.numel() == 0:
        return rbboxes.new_zeros((0, 4))

    cx, cy, w, h, theta = rbboxes.split(1, dim=-1)  # (N,1) each
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # 四个角点（局部坐标）
    dx = w / 2
    dy = h / 2
    # (dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy)
    xs = torch.cat([dx, dx, -dx, -dx], dim=1)  # (N,4)
    ys = torch.cat([dy, -dy, -dy, dy], dim=1)  # (N,4)

    # 旋转 + 平移
    x_rot = cos_t * xs - sin_t * ys + cx
    y_rot = sin_t * xs + cos_t * ys + cy

    x1 = x_rot.min(dim=1, keepdim=True).values
    y1 = y_rot.min(dim=1, keepdim=True).values
    x2 = x_rot.max(dim=1, keepdim=True).values
    y2 = y_rot.max(dim=1, keepdim=True).values

    return torch.cat([x1, y1, x2, y2], dim=1)


@MODELS.register_module()
class SARFPNDenoiser(nn.Module):
    """SAR 场景的 FPN 去噪模块（训练期专用）。

    三类约束：
      1. 船体结构一致性：backbone(C3~C5) vs FPN(P3~P5) 的 RoI 特征 L2
      2. 背景平滑约束：FPN 特征的空间梯度 L1 正则
      3. 角度 / 长宽比一致性：在 FPN RoI 上预测 θ 和 log(w/h)，和 GT 回归

    推理阶段不调用该模块（由 detector 控制），不增加任何推理开销。
    """

    def __init__(self,
                 in_channels_backbone=(256, 512, 1024, 2048),
                 in_channels_fpn=256,
                 backbone_strides=(4, 8, 16, 32),
                 fpn_strides=(8, 16, 32, 64, 128),
                 roi_size=7,
                 ship_geo_weight=0.1,
                 background_smooth_weight=0.01,
                 angle_weight=0.05,
                 aspect_weight=0.05,
                 hidden_dim=256):
        super().__init__()

        self.in_channels_backbone = tuple(in_channels_backbone)
        self.in_channels_fpn = in_channels_fpn
        self.backbone_strides = tuple(backbone_strides)
        self.fpn_strides = tuple(fpn_strides)
        self.roi_size = roi_size

        self.ship_geo_weight = float(ship_geo_weight)
        self.background_smooth_weight = float(background_smooth_weight)
        self.angle_weight = float(angle_weight)
        self.aspect_weight = float(aspect_weight)

        # backbone 各层上的轻量卷积，用于“干净几何参考”投影
        self.backbone_proj = nn.ModuleList()
        for c in self.in_channels_backbone:
            self.backbone_proj.append(
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        # FPN 各层共享的一个卷积“整理”一下特征
        self.fpn_conv = nn.Sequential(
            nn.Conv2d(in_channels_fpn, in_channels_fpn, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_fpn, in_channels_fpn, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 角度 / 长宽比 head：输入为 RoI 池化后的 FPN 向量
        self.angle_head = nn.Sequential(
            nn.Linear(in_channels_fpn, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.aspect_head = nn.Sequential(
            nn.Linear(in_channels_fpn, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    # ----------------- 工具：收集 GT 旋转框 -----------------

    def _gather_gt(self, batch_data_samples, device):
        """从 batch 的 DetDataSample 里收集 rbox + img_ind + angle + log(w/h)."""
        all_rbboxes = []
        all_img_inds = []
        all_angles = []
        all_aspects = []

        for img_id, data_sample in enumerate(batch_data_samples):
            if not hasattr(data_sample, 'gt_instances'):
                continue
            gt_instances = data_sample.gt_instances
            if not hasattr(gt_instances, 'bboxes'):
                continue

            rbboxes = gt_instances.bboxes
            if rbboxes is None:
                continue
            # RotatedBoxes / QuadriBoxes 都有 tensor 属性
            if hasattr(rbboxes, 'tensor'):
                rbboxes = rbboxes.tensor

            if not torch.is_tensor(rbboxes) or rbboxes.numel() == 0:
                continue

            rbboxes = rbboxes.to(device)
            # 期望最后一维是 [cx, cy, w, h, theta]
            if rbboxes.size(-1) < 5:
                continue

            cx, cy, w, h, theta = rbboxes.split(1, dim=-1)
            aspect = torch.log((w + 1e-6) / (h + 1e-6))  # log(w/h)

            num = rbboxes.size(0)
            img_inds = torch.full(
                (num, 1),
                float(img_id),
                dtype=rbboxes.dtype,
                device=device)

            all_rbboxes.append(rbboxes)
            all_img_inds.append(img_inds)
            all_angles.append(theta.view(-1))
            all_aspects.append(aspect.view(-1))

        if len(all_rbboxes) == 0:
            return None

        rbboxes = torch.cat(all_rbboxes, dim=0)
        img_inds = torch.cat(all_img_inds, dim=0)
        angles = torch.cat(all_angles, dim=0)
        aspects = torch.cat(all_aspects, dim=0)
        return rbboxes, img_inds, angles, aspects

    # ----------------- 约束 1：船体结构一致性 -----------------

    def _ship_structure_loss(self,
                             backbone_feats,
                             fpn_feats,
                             rbboxes,
                             img_inds):
        """backbone(C3~C5) & FPN(P3~P5) 的 RoI 向量做 L2 距离。"""
        device = backbone_feats[0].device
        num_backbone = len(backbone_feats)
        num_fpn = len(fpn_feats)

        # C3,C4,C5 对 P3,P4,P5：跳过 C2，对齐前三个 FPN 层
        max_pairs = min(
            num_backbone - 1,
            num_fpn,
            len(self.backbone_strides) - 1,
            len(self.fpn_strides),
        )
        if max_pairs <= 0:
            return backbone_feats[0].sum() * 0.0

        rois_xyxy = obb_to_xyxy(rbboxes)
        # (N, 5): [img_ind, x1, y1, x2, y2]
        rois = torch.cat([img_inds, rois_xyxy], dim=1).to(device)
        if rois.size(0) == 0:
            return backbone_feats[0].sum() * 0.0

        total_loss = 0.0
        valid_levels = 0

        for lvl in range(max_pairs):
            feat_back = backbone_feats[lvl + 1]  # C3,C4,C5
            feat_fpn = fpn_feats[lvl]           # P3,P4,P5

            feat_back_proj = self.backbone_proj[lvl + 1](feat_back)
            feat_fpn_proj = self.fpn_conv(feat_fpn)

            s_back = 1.0 / float(self.backbone_strides[lvl + 1])
            s_fpn = 1.0 / float(self.fpn_strides[lvl])

            # mmcv.ops.roi_align 是 autograd.Function.apply，只能用位置参数
            # 签名大致是：
            # roi_align(input, rois, out_size, spatial_scale, sampling_ratio,
            #           pool_mode='avg', aligned=True)
            roi_back = roi_align(
                feat_back_proj,
                rois,
                self.roi_size,
                s_back,
                0,
                'avg',
                True,
            )
            roi_fpn = roi_align(
                feat_fpn_proj,
                rois,
                self.roi_size,
                s_fpn,
                0,
                'avg',
                True,
            )

            roi_back_vec = F.adaptive_avg_pool2d(roi_back, 1).flatten(1)
            roi_fpn_vec = F.adaptive_avg_pool2d(roi_fpn, 1).flatten(1)

            c = min(roi_back_vec.size(1), roi_fpn_vec.size(1))
            roi_back_vec = roi_back_vec[:, :c]
            roi_fpn_vec = roi_fpn_vec[:, :c]

            loss_lvl = F.mse_loss(roi_fpn_vec, roi_back_vec)
            total_loss = total_loss + loss_lvl
            valid_levels += 1

        if valid_levels == 0:
            return backbone_feats[0].sum() * 0.0
        return total_loss / valid_levels

    # ----------------- 约束 2：背景平滑正则 -----------------

    def _background_smooth_loss(self, fpn_feats):
        """对 FPN 特征做简单的空间梯度 L1 正则（不区分前景/背景）。"""
        if self.background_smooth_weight <= 0:
            return fpn_feats[0].sum() * 0.0

        total = 0.0
        count = 0
        for feat in fpn_feats:
            # feat: (B, C, H, W)
            if feat.size(-1) > 1:
                dx = feat[:, :, :, 1:] - feat[:, :, :, :-1]
                loss_x = dx.abs().mean()
            else:
                loss_x = feat.sum() * 0.0

            if feat.size(-2) > 1:
                dy = feat[:, :, 1:, :] - feat[:, :, :-1, :]
                loss_y = dy.abs().mean()
            else:
                loss_y = feat.sum() * 0.0

            total = total + (loss_x + loss_y)
            count += 1

        if count == 0:
            return fpn_feats[0].sum() * 0.0
        return total / count

    # ----------------- 约束 3：角度 / 长宽比一致性 -----------------

    def _angle_aspect_loss(self,
                           fpn_feats,
                           rbboxes,
                           img_inds,
                           angles,
                           aspects):
        """在多层 FPN RoI 特征上预测 θ 和 log(w/h)，与 GT 回归。"""
        device = fpn_feats[0].device
        rois_xyxy = obb_to_xyxy(rbboxes)
        rois = torch.cat([img_inds, rois_xyxy], dim=1).to(device)
        if rois.size(0) == 0:
            zero = fpn_feats[0].sum() * 0.0
            return zero, zero

        total_angle = 0.0
        total_aspect = 0.0
        valid_levels = 0

        for lvl, stride in enumerate(self.fpn_strides):
            if lvl >= len(fpn_feats):
                break

            feat = fpn_feats[lvl]
            spatial_scale = 1.0 / float(stride)

            roi_feat = roi_align(
                feat,
                rois,
                self.roi_size,
                spatial_scale,
                0,
                'avg',
                True,
            )
            roi_vec = F.adaptive_avg_pool2d(roi_feat, 1).flatten(1)

            pred_angle = self.angle_head(roi_vec).view(-1)
            pred_aspect = self.aspect_head(roi_vec).view(-1)

            target_angle = angles.to(pred_angle.device)
            target_aspect = aspects.to(pred_aspect.device)

            loss_angle = F.smooth_l1_loss(pred_angle, target_angle)
            loss_aspect = F.smooth_l1_loss(pred_aspect, target_aspect)

            total_angle = total_angle + loss_angle
            total_aspect = total_aspect + loss_aspect
            valid_levels += 1

        if valid_levels == 0:
            zero = fpn_feats[0].sum() * 0.0
            return zero, zero

        return total_angle / valid_levels, total_aspect / valid_levels

    # ----------------- 对外接口：给 detector 调 -----------------

    def forward(self, backbone_feats, fpn_feats, batch_data_samples):
        """计算 SAR-aware FPN 去噪相关的 loss 字典."""
        device = backbone_feats[0].device

        # 先算背景平滑（即使没有 GT 也能用）
        bg_smooth = self._background_smooth_loss(fpn_feats)

        gt = self._gather_gt(batch_data_samples, device)
        if gt is None:
            # 没有标注，只能用背景平滑约束
            return dict(
                loss_sar_bg_smooth=self.background_smooth_weight * bg_smooth)

        rbboxes, img_inds, angles, aspects = gt

        ship_geo = self._ship_structure_loss(
            backbone_feats, fpn_feats, rbboxes, img_inds)
        angle_loss, aspect_loss = self._angle_aspect_loss(
            fpn_feats, rbboxes, img_inds, angles, aspects)

        losses = dict(
            loss_sar_ship_geo=self.ship_geo_weight * ship_geo,
            loss_sar_bg_smooth=self.background_smooth_weight * bg_smooth,
            loss_sar_angle=self.angle_weight * angle_loss,
            loss_sar_aspect=self.aspect_weight * aspect_loss,
        )
        return losses
