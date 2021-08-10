_base_ = [
    'ssd_base.py'
]
model = dict(
    detector=dict(
        bbox_head=dict(
            num_classes=1)
    ),
    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None)
)