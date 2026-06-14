custom_imports = dict(
    imports=[
        'mmrotate.models.backbones.attnres_stage_backbone',
        'mmrotate.models.necks.resnet50_pyramid_attn_fusion',
    ],
    allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/ssdd.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'

model = dict(
    type='mmdet.FCOS',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='attnres_stage_backbone',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        use_attnres=True,
        attn_embed_channels=512,
        reduction_ratio=8,
        alpha_init=0.1,
        history_k=1,
        attn_start_stage=1,
        use_gate=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ResNet50_PyramidAttnFusion',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        reduction_ratio=4,
        alpha_init=0.1,
        use_gate=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                        (512, 100000000)),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder',
            angle_version=angle_version),
        angle_coder=dict(
            type='ACMCoder',
            angle_version=angle_version,
            base_omega=2,
            dual_freq=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='RotatedIoULoss',
            loss_weight=1.0),
        loss_angle=dict(
            type='ACMConsistencyLoss',
            beta=0.1111111111111111,
            reg_weight=1.0,
            unit_circle_weight=0.02,
            phase_consistency_weight=0.05,
            loss_weight=0.5),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[28, 33],
        gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='r_coco/bbox_mAP_50',
        rule='greater'))
