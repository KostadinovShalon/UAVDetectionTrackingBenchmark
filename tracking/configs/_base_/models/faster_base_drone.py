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
)