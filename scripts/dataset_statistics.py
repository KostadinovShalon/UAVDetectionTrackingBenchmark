import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
from PIL import Image
from scipy.stats import kde
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from skimage import io, color

args = argparse.ArgumentParser()
args.add_argument("--coco-files", nargs="+", help="List of COCO files path")
args.add_argument("--root-dir", nargs="+", help="List of root directories for each of the coco files. "
                                                "If 1 element is given, it will be used for all of the"
                                                "elements of coco_files")
args.add_argument("--nbins", default=100, type=int, help="Number of bins for the PDF")
opts = args.parse_args()

annotations_file_paths = opts.coco_files
image_root_folders = opts.root_dir
nbins = opts.nbins
x, y, w, h, areas = [], [], [], [], []

assert len(annotations_file_paths) > 0 and len(image_root_folders) > 0
assert len(annotations_file_paths) == len(image_root_folders) or len(image_root_folders) == 1

for ann_file in annotations_file_paths:
    coco = json.load(open(ann_file, 'r'))

    images = coco['images']
    annotations = coco['annotations']

    sizes = {im["id"]: (im["width"], im["height"]) for im in images}

    for ann in annotations:
        bbox = ann['bbox']
        W, H = sizes[ann['image_id']]
        x.append((bbox[0] + bbox[2] / 2) / W)
        y.append((bbox[1] + bbox[3] / 2) / H)
        w.append(bbox[2])
        h.append(bbox[3])
        areas.append(bbox[2] * bbox[3] * 100 / (W * H))

x = np.array(x)
y = 1 - np.array(y)
print(f"Average W, H: {sum(w) / len(w)}, {sum(h) / len(h)}")
print("Average Area:", sum(areas) / len(areas))
areas = np.array(areas)

k = kde.gaussian_kde((x, y))
xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=(4, 4))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='hot')
plt.xlabel("x", fontsize='x-large')
plt.ylabel("y", fontsize='x-large')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.tick_params(axis='both', labelsize='x-large')

# Object dimension statistics:
bins = 100
plt.figure(figsize=(4, 4))
plt.hist(areas, bins, density=True, color='orange')
plt.xlabel("Object area ratio (%)", fontsize='x-large')
plt.ylabel("PDF", fontsize='x-large')
plt.xlim([0, 2.5])
plt.ylim([0, 12.5])
plt.tick_params(axis='both', labelsize='x-large')
plt.show()


# Image Analysis
class StrideSampler(Sampler):
    def __init__(self, data_set, stride):
        self.indices = range(0, len(data_set), stride)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


r, g, b = torch.zeros(256), torch.zeros(256), torch.zeros(256)

for j, ann_file in enumerate(annotations_file_paths):
    dir_index = min(0, j)
    image_root = image_root_folders[dir_index]
    dataset = torchvision.datasets.CocoDetection(image_root, ann_file,
                                                 transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, num_workers=8, sampler=StrideSampler(dataset, 100))
    for i, data in enumerate(tqdm(dataloader)):
        data = data[0]
        r += torch.histc(data[0, 0, ...] * 255, bins=256, min=0, max=255)
        g += torch.histc(data[0, 1, ...] * 255, bins=256, min=0, max=255)
        b += torch.histc(data[0, 2, ...] * 255, bins=256, min=0, max=255)

r = r / r.sum()
g = g / g.sum()
b = b / b.sum()

plt.figure(figsize=(4, 4))
# plt.plot(r.numpy(), color='black')  # Just for IR
plt.plot(r.numpy(), color='red', label='Red')
plt.plot(g.numpy(), color='green', label='Green')
plt.plot(b.numpy(), color='blue', label='Blue')
plt.xlabel("Intensity", fontsize='x-large')
plt.ylabel("PDF", fontsize='x-large')
plt.ylim([0, 0.080])
plt.xlim([0, 255])
plt.tick_params(axis='y', labelsize='large')
plt.tick_params(axis='x', labelsize='x-large')
plt.legend()
plt.show()
