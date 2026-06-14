# Copyright (c) OpenMMLab. All rights reserved.
from .angle_coder import (
    ACMCoder, CDMCoder, CSLCoder, PSCCoder, PseudoAngleCoder, UCResolver)
from .delta_midpointoffset_rbbox_coder import MidpointOffsetCoder
from .delta_xywh_hbbox_coder import DeltaXYWHHBBoxCoder
from .delta_xywh_qbbox_coder import DeltaXYWHQBBoxCoder
from .delta_xywht_hbbox_coder import DeltaXYWHTHBBoxCoder
from .delta_xywht_rbbox_coder import DeltaXYWHTRBBoxCoder
from .distance_angle_point_coder import DistanceAnglePointCoder
from .gliding_vertex_coder import GVFixCoder, GVRatioCoder
from .cf_ucr_angle_coder import CFUCRAngleCoder
#                                             ↑ 这里多加一个
from .cf_ucr_angle_coder import CFUCRAngleCoder


__all__ = [
    'DeltaXYWHTRBBoxCoder', 'DeltaXYWHTHBBoxCoder', 'MidpointOffsetCoder',
    'GVFixCoder', 'GVRatioCoder', 'CSLCoder', 'PSCCoder', 'ACMCoder',
    'CDMCoder', 'UCResolver',
    'DistanceAnglePointCoder', 'DeltaXYWHHBBoxCoder', 'DeltaXYWHQBBoxCoder',
    'PseudoAngleCoder', 'CFUCRAngleCoder'
]
