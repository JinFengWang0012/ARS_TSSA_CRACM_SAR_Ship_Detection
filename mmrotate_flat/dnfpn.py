import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List

from mmengine.model import BaseModule, constant_init, xavier_init

from mmrotate.registry import MODELS
from mmdet.models.necks import FPN
from mmdet.registry import MODELS as MMDET_MODELS


class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling) for DNFPN.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels for each ASPP branch.
        dilations (Sequence[int]): Dilation rates for branches. The last
            branch uses global average pooling.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilations: Sequence[int],
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations

        self.aspp = nn.ModuleList()
        for dilation in dilations:
            # DetectoRS 实现里: dilation>1 用 3×3 空洞卷积, 否则 1×1
            if dilation > 1:
                kernel_size = 3
                padding = dilation
            else:
                kernel_size = 1
                padding = 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)

        # 全局平均池化分支
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最后一支用 GAP 的输入，其余用原 feature
        avg_x = self.gap(x)
        outs = []
        for idx, conv in enumerate(self.aspp):
            inp = avg_x if (idx == len(self.aspp) - 1) else x
            out = F.relu_(conv(inp))
            outs.append(out)

        # GAP 分支的输出扩展到和其他分支同 spatial size
        outs[-1] = outs[-1].expand_as(outs[-2])
        out = torch.cat(outs, dim=1)
        return out


@MODELS.register_module()
class DNFPN(FPN):
    """DeNoising FPN (DNFPN) from DNTR.

    本质上是 DetectoRS 的 RFP:
      - 先做一次普通 FPN
      - 对高层特征做 ASPP 得到 rfp_feats
      - 用 DetectoRS_ResNet 的 rfp_forward(img, rfp_feats) 再跑一遍 backbone
      - 再做一次 FPN
      - 用 1×1 conv 预测权重，对新旧 FPN 特征做加权融合
      - 如此递归 rfp_steps-1 次

    注意:
        forward 的 inputs 需要是 (img, C2, C3, C4, C5, ...) 这样的形式，
        第 0 个元素是原始图像张量。
    """

    def __init__(self,
                 rfp_steps: int,
                 rfp_backbone: dict,
                 aspp_out_channels: int,
                 aspp_dilations: Sequence[int] = (1, 3, 6, 1),
                 init_cfg=None,
                 **kwargs) -> None:
        # 为了和原实现对齐，这里不允许外部再传 init_cfg
        assert init_cfg is None, (
            'To prevent abnormal initialization behavior, '
            'init_cfg is not allowed to be set for DNFPN.')
        super().__init__(init_cfg=None, **kwargs)

        self.rfp_steps = rfp_steps
        assert self.rfp_steps >= 1

        # 构建递归用的 backbone 模块 (DetectoRS_ResNet 等)
        self.rfp_modules = nn.ModuleList()
        for _ in range(1, rfp_steps):
            rfp_module = MMDET_MODELS.build(rfp_backbone)
            self.rfp_modules.append(rfp_module)

        # ASPP: 输入是 FPN 的 out_channels
        self.rfp_aspp = ASPP(
            in_channels=self.out_channels,
            out_channels=aspp_out_channels,
            dilations=aspp_dilations)

        # 用于去噪的 per-pixel 权重 conv
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def init_weights(self) -> None:
        """Initialize DNFPN weights.

        不调用 super().init_weights()，避免对 rfp_modules 的 init_cfg
        产生奇怪的影响，保持和 DNTR / DetectoRS 行为一致。
        """
        # 初始化 lateral_convs 和 fpn_convs
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')

        # 初始化 RFP 用的 backbone
        for rfp_module in self.rfp_modules:
            if hasattr(rfp_module, 'init_weights'):
                rfp_module.init_weights()

        # rfp_weight 初始化为 0，让初始阶段更偏向于“保留旧特征”
        constant_init(self.rfp_weight, 0)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Forward function.

        Args:
            inputs: tuple of (img, C2, C3, C4, C5, ...)

        Returns:
            Tuple[Tensor]: 多尺度 FPN 特征.
        """
        inputs = list(inputs)
        # +1 是因为 inputs[0] 是原图
        assert len(inputs) == len(self.in_channels) + 1, \
            f'Expect {len(self.in_channels)+1} inputs (img + feats), ' \
            f'but got {len(inputs)}'

        img = inputs.pop(0)

        # 先跑一遍标准 FPN
        x = super().forward(tuple(inputs))  # List[Tensor]

        # 递归 rfp_steps-1 次
        for rfp_idx in range(self.rfp_steps - 1):
            # 构造给 rfp_backbone 的多尺度输入:
            # 第一层用原 FPN 低层特征，其余用 ASPP 过的高层特征
            rfp_feats: List[torch.Tensor] = [x[0]]
            for i in range(1, len(x)):
                rfp_feats.append(self.rfp_aspp(x[i]))

            # 走 DetectoRS_ResNet 的 rfp_forward(img, rfp_feats)
            rfp_backbone = self.rfp_modules[rfp_idx]
            if not hasattr(rfp_backbone, 'rfp_forward'):
                raise RuntimeError(
                    'rfp_backbone must implement `rfp_forward(img, rfp_feats)` '
                    'to be used in DNFPN.')

            x_idx = rfp_backbone.rfp_forward(img, tuple(rfp_feats))

            # 再跑一遍 FPN
            x_idx = super().forward(x_idx)

            # 用 1×1 conv 预测 per-pixel 权重，对新旧 FPN 特征做加权融合
            x_new: List[torch.Tensor] = []
            for ft_old, ft_new in zip(x, x_idx):
                add_weight = torch.sigmoid(self.rfp_weight(ft_new))
                x_new.append(add_weight * ft_new + (1.0 - add_weight) * ft_old)
            x = x_new

        return tuple(x)
