input_size = 512
model = dict(
    detector=dict(
        type='SingleStageDetector',
        pretrained='open-mmlab://vgg16_caffe',
        neck=None,
        backbone=dict(
            type='SSDVGG',
            input_size=input_size,
            depth=16,
            with_last_pool=False,
            ceil_mode=True,
            out_indices=(3, 4),
            out_feature_indices=(22, 34),
            l2_norm_scale=20),
        bbox_head=dict(
            type='SSDHead',
            in_channels=(512, 1024, 512, 256, 256, 256, 256),
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=input_size,
                basesize_ratio_range=(0.1, 0.9),
                strides=[8, 16, 32, 64, 128, 256, 512],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])),
        # bbox_coder=dict(clip_border=False)))
    ))
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
        # transforms=[
        #     dict(type='Resize', keep_ratio=False),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='ImageToTensor', keys=['img']),
        #     dict(type='Collect', keys=['img']),
        # ]
    )
]