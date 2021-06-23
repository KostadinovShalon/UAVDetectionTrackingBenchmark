_base_ = [
    'yolo_base.py'
]
model = dict(
    detector=dict(
        bbox_head=dict(
            num_classes=1)
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
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
        num_frames_retain=100),
)