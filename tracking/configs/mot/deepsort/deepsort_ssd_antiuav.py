_base_ = [
    'ssd_merged.py',
    '../../_base_/datasets/mot_challenge_ssd_antiuav.py', '../../_base_/default_runtime.py'
]
dataset_type = 'CocoVideoDataset'
classes = ('drone',)
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../mmtracking/data/anti-uav/images/anti-uav-full.json',
        img_prefix='../mmtracking/data/anti-uav/images/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../mmtracking/data/anti-uav/images/anti-uav-test-full.json',
        img_prefix='../mmtracking/data/anti-uav/images/'
        ))
model = dict(
    type='DeepSORT',
    pretrains=dict(
        detector=  # noqa: E251
        # 'C:\\Users\\Matt\\Documents\\PhD\\drone\\mmtracking\\configs\\mot\\deepsort\\faster.pth', # noqa: E501
        "/home2/lgfm95/drone/Strig-UAV-Project/anti-uav/SSD512/full/latest.pth",
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ),
    # detector=dict(
    # type='SingleStageDetector',
    # pretrained='open-mmlab://vgg16_caffe',
    # backbone=dict(
    #     type='SSDVGG',
    #     input_size=512,
    #     depth=16,
    #     with_last_pool=False,
    #     ceil_mode=True,
    #     out_indices=(3, 4),
    #     out_feature_indices=(22, 34),
    #     l2_norm_scale=20),
    # neck=None,
    # bbox_head=dict(
    #     type='SSDHead',
    #     in_channels=(512, 1024, 512, 256, 256, 256, 256),
    #     num_classes=1,
    #     anchor_generator=dict(
    #         type='SSDAnchorGenerator',
    #         scale_major=False,
    #         input_size=512,
    #         basesize_ratio_range=(0.1, 0.9),
    #         strides=[8, 16, 32, 64, 128, 256, 512],
    #         ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[0.0, 0.0, 0.0, 0.0],
    #         target_stds=[0.1, 0.1, 0.2, 0.2]))),

    detector=dict(
        bbox_head=dict(
            num_classes=1),
        test_cfg=dict(
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200)),
    # motion=dict(type='CondiFilter', center_only=False),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))),
    tracker=dict(
        # type='CondiTracker',
        type='SortTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
