_base_ = [
    'yolo_base.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py']
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
    type='Tracktor',
    pretrains=dict(
        detector=  # noqa: E251
        # 'C:\\Users\\Matt\\Documents\\PhD\\drone\\mmtracking\\configs\\mot\\deepsort\\faster.pth', # noqa: E501
        "/home2/lgfm95/drone/mmtracking/configs/mot/deepsort/yolo_bird.pth",
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ),
    detector=dict(
    bbox_head=dict(
        num_classes=1)),
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
