_base_ = [
    'faster_rcnn_r50_fpn.py'
]
model = dict(
    detector=dict(
        roi_head=dict(
            bbox_head=dict(
                num_classes=1)
        )
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None)
)