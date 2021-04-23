import os

import matplotlib.pyplot as plt
import torch
import torchvision.datasets
from skimage import io, color
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

annotations_file_paths = \
    {
        'Drone-vs-Bird': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/drone-vs-bird/train.json',
            '/home/brian/Documents/PhD/UAV detection/Datasets/drone-vs-bird/val.json'
        ],
        'MAV-VID': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/mav_vid_dataset/train/annotations.json',
            '/home/brian/Documents/PhD/UAV detection/Datasets/mav_vid_dataset/val/annotations.json',
        ],
        'Anti-UAV RGB': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/anti-uav-rgb.json',
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/anti-uav-test-rgb.json'
        ],
        'Anti-UAV IR': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/anti-uav-ir.json',
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/anti-uav-test-ir.json'
        ]
    }

image_root_folders = \
    {
        'Drone-vs-Bird': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/drone-vs-bird/images/drone-vs-bird',
            '/home/brian/Documents/PhD/UAV detection/Datasets/drone-vs-bird/images/drone-vs-bird'
        ],
        'MAV-VID': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/mav_vid_dataset/train/img',
            '/home/brian/Documents/PhD/UAV detection/Datasets/mav_vid_dataset/val/img',
        ],
        'Anti-UAV RGB': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/test-dev',
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/test-dev'
        ],
        'Anti-UAV IR': [
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/test-dev',
            '/home/brian/Documents/PhD/UAV detection/Datasets/anti-uav/images/test-dev'
        ]
    }
# Image Analysis


class StrideSampler(Sampler):
    def __init__(self, data_set, stride):
        self.indices = range(0, len(data_set), stride)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class LABCocoDetection(torchvision.datasets.CocoDetection):

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = io.imread(os.path.join(self.root, path))
        img = color.rgb2lab(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


lums = {}
for k in image_root_folders.keys():
    lum = torch.zeros(100)

    for ann_file, image_root in zip(annotations_file_paths[k], image_root_folders[k]):
        dataset = LABCocoDetection(image_root, ann_file, transform=torchvision.transforms.ToTensor())
        dataloader = DataLoader(dataset, num_workers=8, sampler=StrideSampler(dataset, 100))
        for i, data in enumerate(tqdm(dataloader)):
            data = data[0]
            lum += torch.histc(data[0, 0, ...], bins=100, min=0, max=100)

    lums[k] = lum = lum / lum.sum()


plt.figure(figsize=(4, 4))
for k, v in lums.items():
    plt.plot(v.numpy(), label=k)

plt.xlim([0, 100])
plt.xlabel("Luminance", fontsize='x-large')
plt.ylabel("PDF", fontsize='x-large')
plt.tick_params(axis='y', labelsize='large')
plt.tick_params(axis='x', labelsize='x-large')
plt.legend()
plt.show()