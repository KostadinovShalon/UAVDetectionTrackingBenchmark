_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']
dataset_type = 'CocoVideoDataset'
classes = ('drone',)
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../mmtracking/data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/annotations.json',
        img_prefix='../mmtracking/data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/img/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../mmtracking/data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/annotations.json',
        # ann_file='data/strig-drones/datasets/mav-vid/val_annotations.json',
        img_prefix='../mmtracking/data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/img/'
        ))
model = dict(
    type='Tracktor',
    pretrains=dict(
        detector=  # noqa: E251
        # 'C:\\Users\\Matt\\Documents\\PhD\\drone\\mmtracking\\configs\\mot\\deepsort\\faster.pth', # noqa: E501
        "/home2/lgfm95/drone/Strig-UAV-Project/MAV-VID/fasterrcnn/latest.pth",
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ),
    detector=dict(
        type='FasterRCNN',
        pretrained='torchvision://resnet50',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch'),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)))),
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
    motion=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    tracker=dict(
        type='TracktorTracker',
        obj_score_thr=0.5,
        regression=dict(
            obj_score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.6),
            match_iou_thr=0.3),
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0,
            match_iou_thr=0.2),
        momentums=None,
        num_frames_retain=10))
# data_root = 'data/MOT17/'
# test_set = 'train'
# data = dict(
#     train=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
#     val=dict(ann_file=data_root + 'annotations/train_cocoformat.json'),
#     test=dict(
#         ann_file=data_root + f'annotations/{test_set}_cocoformat.json',
#         img_prefix=data_root + test_set))
