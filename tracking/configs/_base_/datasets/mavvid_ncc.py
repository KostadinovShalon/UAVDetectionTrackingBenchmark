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
        ann_file='../data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/annotations.json',
        img_prefix='../data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/train/img/'
        # ann_file='data/strig-drones/datasets/mav-vid/train_annotations.json',
        ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/annotations.json',
        # ann_file='data/strig-drones/datasets/mav-vid/val_annotations.json',
        img_prefix='../data/multirotor-aerial-vehicle-vid-mavvid-dataset/mav_vid_dataset/val/img/'
        ))