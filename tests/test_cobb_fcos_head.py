import unittest

import pytest
import torch

from mmrotate.models.dense_heads import COBBFCOSHead
from mmrotate.structures import RotatedBoxes
from mmrotate.structures.bbox import rbbox_overlaps
from mmrotate.utils import register_all_modules


class TestCOBBFCOSHead(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def test_rs_rln_inverse(self):
        head = COBBFCOSHead(
            num_classes=1,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)
        rs = torch.tensor([0.08, 0.15, 0.22, 0.35, 0.49], dtype=torch.float32)
        area_ratio = torch.tensor([0.30, 0.35, 0.45, 0.60, 0.75],
                                  dtype=torch.float32)
        rln = head._rs_to_rln(rs, area_ratio)
        restored = head._invert_rln(rln)
        self.assertTrue(torch.allclose(restored, rs, atol=1e-5, rtol=1e-4))

    @pytest.mark.skipif(
        not hasattr(torch.ops, 'mmcv'),
        reason='mmcv ops are required for rotated IoU validation')
    def test_cobb_roundtrip(self):
        head = COBBFCOSHead(
            num_classes=1,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)
        target_rboxes = torch.tensor(
            [[64.0, 64.0, 40.0, 12.0, 0.25],
             [80.0, 48.0, 18.0, 42.0, -0.62],
             [128.0, 96.0, 32.0, 20.0, 1.10]],
            dtype=torch.float32)
        points = target_rboxes[:, :2].clone()
        strides = torch.full((target_rboxes.size(0), 1), 8.0)

        decoded, reg_targets, style_inds = head.roundtrip_rboxes(
            target_rboxes, points, strides)

        self.assertEqual(decoded.shape, target_rboxes.shape)
        self.assertEqual(reg_targets.shape, (target_rboxes.size(0), 5))
        self.assertEqual(style_inds.shape, (target_rboxes.size(0), ))

        overlaps = rbbox_overlaps(decoded, target_rboxes, is_aligned=True)
        self.assertTrue(torch.all(overlaps > 0.95),
                        msg=f'Roundtrip IoUs too low: {overlaps.tolist()}')

        ctr_error = (decoded[:, :2] - target_rboxes[:, :2]).abs().max()
        self.assertLess(float(ctr_error), 1e-4)
