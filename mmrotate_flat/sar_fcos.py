# mmrotate/models/detectors/sar_fcos.py
import torch
from mmdet.models import FCOS
from mmrotate.registry import MODELS


@MODELS.register_module()
class SARFPNDNFCOS(FCOS):
    """带 SAR FPN 去噪模块的 FCOS

    - 结构上完全复用 mmdet.FCOS + RotatedFCOSHead
    - 训练时：多算一个 SARFPNDenoiser 的 loss
    - 测试时：不调用去噪模块，不增加任何 FLOPs
    """

    def __init__(self, sar_denoiser=None, **kwargs):
        super().__init__(**kwargs)
        if sar_denoiser is not None:
            self.sar_denoiser = MODELS.build(sar_denoiser)
        else:
            self.sar_denoiser = None
        self._sar_cache = None

    def extract_feat(self, batch_inputs: torch.Tensor):
        """重载 extract_feat，把 backbone/FPN 特征缓存下来给去噪模块用。"""
        backbone_feats = self.backbone(batch_inputs)
        if self.with_neck:
            fpn_feats = self.neck(backbone_feats)
        else:
            fpn_feats = backbone_feats

        if self.training and self.sar_denoiser is not None:
            self._sar_cache = dict(
                backbone_feats=backbone_feats,
                fpn_feats=fpn_feats,
            )
        else:
            self._sar_cache = None

        return fpn_feats

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """在原 FCOS 的 loss 基础上，加 SAR 去噪 loss."""
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

        if self.training and self.sar_denoiser is not None \
                and self._sar_cache is not None:
            sar_losses = self.sar_denoiser(
                self._sar_cache['backbone_feats'],
                self._sar_cache['fpn_feats'],
                batch_data_samples)
            losses.update(sar_losses)

        return losses
