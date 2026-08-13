# Copyright (c) OpenMMLab. All rights reserved.
from .angle_branch_retina_head import AngleBranchRetinaHead
# from .acmangle_branch_retina_head import ACMAngleBranchRetinaHead
from .cfa_head import CFAHead
from .h2rbox_head import H2RBoxHead
from .h2rbox_v2_head import H2RBoxV2Head
from .oriented_reppoints_head import OrientedRepPointsHead
from .oriented_rpn_head import OrientedRPNHead
from .r3_head import R3Head, R3RefineHead
from .rotated_atss_head import RotatedATSSHead
from .continuous_refine_rotated_fcos_head import ContinuousRefineRotatedFCOSHead
from .cobb_fcos_head import COBBFCOSHead
from .decoupled_angle_aware_rotated_fcos_head import \
    DecoupledAngleAwareRotatedFCOSHead
from .gaucho_fcos_head import GauChoFCOSHead
from .gaucho_ap50ap75_fcos_head import GauChoAP50AP75FCOSHead
from .geo_aware_gaucho_ap50ap75_fcos_head import (
    GeometryAwareGauChoAP50AP75FCOSHead)
from .gaucho_paper_fcos_head import GauChoPaperFCOSHead
from .rotated_fcos_head import RotatedFCOSHead
from .rotated_fcos_gaussian_quality_head import RotatedFCOSGaussianQualityHead
from .rotated_fcos_gd_aux_head import RotatedFCOSGDAuxHead
from .rotated_fcos_pqa_head import RotatedFCOSPQAHead
from .rotated_reppoints_head import RotatedRepPointsHead
from .rotated_retina_head import RotatedRetinaHead
from .rotated_rtmdet_head import RotatedRTMDetHead, RotatedRTMDetSepBNHead
from .s2a_head import S2AHead, S2ARefineHead
from .sam_reppoints_head import SAMRepPointsHead
from .CDangle_branch_retina_head import CDAngleBranchRetinaHead
# from .regloss_no_cracmangle_branch_retina_head import REGLOSSNOCRACMAngleBranchRetinaHead
__all__ = [
    'RotatedRetinaHead', 'OrientedRPNHead', 'RotatedRepPointsHead',
    'SAMRepPointsHead', 'AngleBranchRetinaHead', 'RotatedATSSHead',
    'RotatedFCOSHead', 'OrientedRepPointsHead', 'R3Head', 'R3RefineHead',
    'S2AHead', 'S2ARefineHead', 'CFAHead', 'H2RBoxHead', 'H2RBoxV2Head',
    'RotatedRTMDetHead', 'RotatedRTMDetSepBNHead',
    'CDAngleBranchRetinaHead', 'ContinuousRefineRotatedFCOSHead',
    'COBBFCOSHead', 'DecoupledAngleAwareRotatedFCOSHead', 'GauChoFCOSHead',
    'GauChoAP50AP75FCOSHead',
    'GeometryAwareGauChoAP50AP75FCOSHead', 'GauChoPaperFCOSHead',
    'RotatedFCOSGaussianQualityHead', 'RotatedFCOSGDAuxHead',
    'RotatedFCOSPQAHead']
