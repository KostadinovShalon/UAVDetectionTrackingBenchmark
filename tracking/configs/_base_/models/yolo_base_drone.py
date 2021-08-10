_base_ = [
    'yolo_base.py'
]
model = dict(
    detector=dict(
        bbox_head=dict(
            num_classes=1)
    ),
)