_base_ = [
    'ssd300.py',
]
input_size = 512
model = dict(
    detector=dict(
        backbone=dict(input_size=input_size),
        bbox_head=dict(
            in_channels=(512, 1024, 512, 256, 256, 256, 256),
            num_classes=1,
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=input_size,
                basesize_ratio_range=(0.1, 0.9),
                strides=[8, 16, 32, 64, 128, 256, 512],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
            bbox_coder=dict(clip_border=False)
        )
    ),
)
