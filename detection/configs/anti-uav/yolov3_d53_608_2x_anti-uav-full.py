_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
# model settings
model = dict(
    bbox_head=dict(
        num_classes=1))
checkpoint_config = dict(interval=3)
# dataset settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# Modify dataset related settings
classes = ('drone',)
data = dict(
    samples_per_gpu=5,
    train=dict(
        img_prefix='data/anti-uav/images/',
        classes=classes,
        ann_file='data/anti-uav/train-full.json'),
    val=dict(
        img_prefix='data/anti-uav/images/',
        classes=classes,
        ann_file='data/anti-uav/val-full.json'),
    test=dict(
        img_prefix='data/anti-uav/images/',
        classes=classes,
        ann_file='data/anti-uav/val-full.json'))

load_from = 'checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
# lr_config = dict(
#     step=[75, 90])
# # runtime settings
# total_epochs = 100
evaluation = dict(interval=1, metric=['bbox'])

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24

