_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
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
        ann_file='data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/annotations.json',
        img_prefix='data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/img/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/annotations.json',
        # ann_file='data/strig-drones/datasets/mav-vid/val_annotations.json',
        img_prefix='data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/img/'
        ))
model = dict(
    type='DeepSORT',
    pretrains=dict(
        detector=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth',  # noqa: E501
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ),
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(
                clip_border=False), num_classes=1))),
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
