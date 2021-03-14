_base_ = '../detr/detr_r50_8x2_150e_coco.py'
model = dict(bbox_head=dict(num_classes=1,))
classes = ('drone',)
checkpoint_config = dict(interval=3)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='data/mav_vid_dataset/images/',
        classes=classes,
        ann_file='data/mav_vid_dataset/train.json'),
    val=dict(
        img_prefix='data/mav_vid_dataset/images/',
        classes=classes,
        ann_file='data/mav_vid_dataset/val.json'),
    test=dict(
        img_prefix='data/mav_vid_dataset/images/',
        classes=classes,
        ann_file='data/mav_vid_dataset/val.json'))

load_from = 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24
