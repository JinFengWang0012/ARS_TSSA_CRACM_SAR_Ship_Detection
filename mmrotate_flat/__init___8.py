# Copyright (c) OpenMMLab. All rights reserved.
from .re_resnet import ReResNet
from .arc_resnet import ARCResNet
from .my_backbone import MyBackbone
from .softmax_mybackbone import SoftmaxMyBackbone
from .hybrid_history_backbone import HybridHistoryBackbone
from .gated_history_backbone import GatedHistoryBackbone

from .attnres_stage_backbone import AttnResStageBackbone

from .attnres_pyramid_backbone import AttnResPyramidBackbone

__all__ = [
    'ReResNet', 'ARCResNet', 'MyBackbone', 'SoftmaxMyBackbone',
    'HybridHistoryBackbone', 'GatedHistoryBackbone',
     'AttnResStageBackbone',
    
]



