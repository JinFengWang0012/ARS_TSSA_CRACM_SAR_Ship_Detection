import argparse

import torch

from mmrotate.models.dense_heads import COBBFCOSHead
from mmrotate.structures.bbox import rbbox_overlaps
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Debug COBB encode/decode roundtrip quality.')
    parser.add_argument(
        '--stride',
        type=float,
        default=8.0,
        help='Stride used to normalize COBB regression targets.')
    return parser.parse_args()


def main():
    args = parse_args()
    register_all_modules()

    head = COBBFCOSHead(
        num_classes=1,
        in_channels=1,
        feat_channels=1,
        stacked_convs=1,
        norm_cfg=None)

    target_rboxes = torch.tensor(
        [[64.0, 64.0, 40.0, 12.0, 0.25],
         [80.0, 48.0, 18.0, 42.0, -0.62],
         [128.0, 96.0, 32.0, 20.0, 1.10],
         [144.0, 144.0, 56.0, 10.0, -1.20]],
        dtype=torch.float32)
    points = target_rboxes[:, :2].clone()
    strides = torch.full((target_rboxes.size(0), 1), float(args.stride))

    decoded, reg_targets, style_inds = head.roundtrip_rboxes(
        target_rboxes, points, strides)
    overlaps = rbbox_overlaps(decoded, target_rboxes, is_aligned=True)

    print('COBB roundtrip debug')
    print(f'stride={args.stride}')
    print('style_inds=', style_inds.tolist())
    print('reg_targets=')
    print(reg_targets)
    print('decoded=')
    print(decoded)
    print('gt=')
    print(target_rboxes)
    print('aligned_iou=')
    print(overlaps)
    print('min_iou=', float(overlaps.min()))
    print('mean_iou=', float(overlaps.mean()))


if __name__ == '__main__':
    main()
