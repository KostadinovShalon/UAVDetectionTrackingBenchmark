_base_ = '../ssd/ssd512_coco.py'
# model settings
model = dict(
    bbox_head=dict(
        num_classes=1))
checkpoint_config = dict(interval=3)
# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
classes = ('drone',)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/drone-vs-bird/images/',
        classes=classes,
        ann_file='data/drone-vs-bird/train.json',
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/drone-vs-bird/images/',
        classes=classes,
        ann_file='data/drone-vs-bird/val.json',
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type=dataset_type,
        img_prefix='data/drone-vs-bird/images/',
        classes=classes,
        ann_file='data/drone-vs-bird/val.json',
        pipeline=test_pipeline))

load_from = 'checkpoints/ssd512_coco_20200308-038c5591.pth'
