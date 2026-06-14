from mmrotate.registry import MODELS
from .rotated_fcos_head import RotatedFCOSHead


@MODELS.register_module()
class ContinuousRefineRotatedFCOSHead(RotatedFCOSHead):
    """Compatibility wrapper for an RSAR-style Rotated FCOS head.

    The old coarse-to-fine residual angle branch is intentionally removed.
    Angle learning is handled by the configured continuous angle coder,
    typically ``UCResolver``, plus the base ``RotatedFCOSHead`` losses.
    """
