_base_ = [
    'yolo_base.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
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
        ann_file='../mmtracking/data/birdvsdrone/train.json',
        img_prefix='../mmtracking/data/birdvsdrone/images/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../mmtracking/data/birdvsdrone/train.json',
        img_prefix='../mmtracking/data/birdvsdrone/images/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ))
model = dict(
    type='DeepSORT',
    pretrains=dict(
        detector=  # noqa: E251
        # 'C:\\Users\\Matt\\Documents\\PhD\\drone\\mmtracking\\configs\\mot\\deepsort\\faster.pth', # noqa: E501
        "/home2/lgfm95/drone/Strig-UAV-Project/bird-vs-drone/YOLOv3/latest.pth",
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ),
    # detector=dict(
    # type='YOLOV3',
    # pretrained='open-mmlab://darknet53',
    # backbone=dict(type='Darknet', depth=53, out_indices=(3, 4, 5)),
    # neck=dict(
    #     type='YOLOV3Neck',
    #     num_scales=3,
    #     in_channels=[1024, 512, 256],
    #     out_channels=[512, 256, 128]),
    # bbox_head=dict(
    #     type='YOLOV3Head',
    #     num_classes=1,
    #     in_channels=[512, 256, 128],
    #     out_channels=[1024, 512, 256],
    #     anchor_generator=dict(
    #         type='YOLOAnchorGenerator',
    #         base_sizes=[[(116, 90), (156, 198), (373, 326)],
    #                     [(30, 61), (62, 45), (59, 119)],
    #                     [(10, 13), (16, 30), (33, 23)]],
    #         strides=[32, 16, 8]),
    #     bbox_coder=dict(type='YOLOBBoxCoder'),
    #     featmap_strides=[32, 16, 8],
    #     loss_cls=dict(
    #         type='CrossEntropyLoss',
    #         use_sigmoid=True,
    #         loss_weight=1.0,
    #         reduction='sum'),
    #     loss_conf=dict(
    #         type='CrossEntropyLoss',
    #         use_sigmoid=True,
    #         loss_weight=1.0,
    #         reduction='sum'),
    #     loss_xy=dict(
    #         type='CrossEntropyLoss',
    #         use_sigmoid=True,
    #         loss_weight=2.0,
    #         reduction='sum'),
    #     loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum'))),

    detector=dict(
    bbox_head=dict(
        num_classes=1)),
    # motion=dict(type='CondiFilter', center_only=False),
    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None))
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