_base_ = [
    'mot_challenge.py'
]
dataset_type = 'CocoVideoDataset'
classes = ('drone',)
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/birdvsdrone/train.json',
        img_prefix='../data/birdvsdrone/images/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/birdvsdrone/train.json',
        img_prefix='../data/birdvsdrone/images/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ))