# mmrotate/models/necks/dnfpn.py

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmdet.models.necks import FPN
from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType


@MODELS.register_module()
class DNFPN(FPN):
    """DeNoising FPN (DN-FPN) from
    'A DeNoising FPN with Transformer R-CNN for Tiny Object Detection'.

    核心设计：
    - 结构上仍然是一个标准 FPN（继承 mmdet.FPN）
    - 训练阶段额外对 FPN 的特征做两类编码：
        * 几何编码（geometric encoder）：更贴近 bottom-up / lateral 特征
        * 语义编码（semantic encoder）：更贴近 top-down / fusion 特征
    - 然后用对比学习损失，让融合特征在“几何”和“语义”上都和对应源特征保持一致
    - 推理阶段仅使用 FPN 的输出，不额外增加算力

    注意：
    - 这里只提供结构和接口骨架，具体的对比损失公式需要你参考论文实现：
        L_geo, L_sem，最后在 detector.loss() 里调用 get_dnfpn_loss()。
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 # FPN 自带参数
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: bool = False,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 # DN-FPN 自己的参数
                 geo_embed_channels: int = 128,
                 sem_embed_channels: int = 128,
                 dn_loss_weight: float = 1.0,
                 init_cfg: OptConfigType = dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform')):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        self.geo_embed_channels = geo_embed_channels
        self.sem_embed_channels = sem_embed_channels
        self.dn_loss_weight = dn_loss_weight

        # 简单几何编码器：对 lateral 特征做全局池化 + FC
        # 你可以按论文改成更复杂的 encoder
        self.geo_encoders = nn.ModuleList()
        # 简单语义编码器：对 top-down / fusion 特征做全局池化 + FC
        self.sem_encoders = nn.ModuleList()

        num_in_levels = len(self.in_channels)
        for _ in range(num_in_levels):
            self.geo_encoders.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.out_channels, geo_embed_channels, 1),
                    nn.ReLU(inplace=True)
                ))
            self.sem_encoders.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.out_channels, sem_embed_channels, 1),
                    nn.ReLU(inplace=True)
                ))

        # 这里不在 __init__ 里 build loss，直接在 get_dnfpn_loss 里用 F.cosine_embedding_loss 等
        # 或者你可以仿照 mmdet 的 build_loss 来封装

        # 存训练阶段用的中间 embedding
        self._latest_geo_embeds: Optional[List[Tensor]] = None
        self._latest_sem_embeds: Optional[List[Tensor]] = None

    def forward(self, inputs: List[Tensor]) -> Tuple[List[Tensor]]:
        """前向和 FPN 一样返回多尺度特征。

        同时在训练阶段额外保存几何/语义 embedding，
        供 get_dnfpn_loss() 在 detector.loss() 中调用。
        """
        # FPN 正常前向
        outs = super().forward(inputs)

        if self.training:
            # 注意：标准 FPN 里我们拿不到“裸的 lateral/上采样中间结果”，
            # 这里用 outs 作为语义特征，用 backbone 的 inputs 作为几何特征近似处理。
            # 如果你想完全对齐论文，可以参考 mmdet 源码，把 lateral_feat 暴露出来。
            geo_embeds = []
            sem_embeds = []

            # 几何信息更多在浅层，所以这里用 backbone 的 inputs 作为几何源特征
            for i, x in enumerate(inputs):
                # 先把通道映射到 out_channels（如果通道数不匹配可以加个 1x1 conv，这里简化略过）
                if x.size(1) != self.out_channels:
                    # 简单用全局池化 + 线性层表示几何 embedding
                    pooled = x.mean(dim=[2, 3], keepdim=True)
                    embed = nn.functional.relu(pooled)
                else:
                    embed = self.geo_encoders[min(i, len(self.geo_encoders)-1)](x)
                geo_embeds.append(embed.squeeze(-1).squeeze(-1))

            # 语义信息更多在 FPN 输出的各个层 outs
            for i, p in enumerate(outs):
                embed = self.sem_encoders[min(i, len(self.sem_encoders)-1)](p)
                sem_embeds.append(embed.squeeze(-1).squeeze(-1))

            self._latest_geo_embeds = geo_embeds
            self._latest_sem_embeds = sem_embeds

        return outs

    def get_dnfpn_loss(self) -> dict:
        """根据最近一次 forward 保存的 embedding 计算 DN-FPN 的对比损失。

        这里给你一个非常简化的实现示例：
        - 对应层的几何 embedding 和语义 embedding 应该“接近”
        - 不同层之间可以视为负样本（如果你想做完整对比学习，可以扩展）

        真正要对齐论文：请用 InfoNCE / NT-Xent 等对比损失，
        并按论文定义的 L_geo、L_sem 去实现。
        """
        if (self._latest_geo_embeds is None or
                self._latest_sem_embeds is None):
            return dict()

        geo_embeds = self._latest_geo_embeds
        sem_embeds = self._latest_sem_embeds

        # 简单用 MSE 做“几何-语义一致性”示例
        loss_geo = 0.0
        loss_sem = 0.0
        num_levels = min(len(geo_embeds), len(sem_embeds))
        for i in range(num_levels):
            g = geo_embeds[i]
            s = sem_embeds[i]
            # L2 一致性损失（示例）
            loss_geo = loss_geo + nn.functional.mse_loss(g, s)

        # 这里只返回一个合并损失，你也可以拆成 loss_geo / loss_sem
        loss_dnfpn = self.dn_loss_weight * loss_geo

        # 清空，避免跨 batch 累积
        self._latest_geo_embeds = None
        self._latest_sem_embeds = None

        return dict(loss_dnfpn=loss_dnfpn)
