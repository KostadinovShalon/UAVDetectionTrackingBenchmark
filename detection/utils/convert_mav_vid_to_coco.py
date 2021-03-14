#####################################################################

# Gets the MAV-VID yolo annotations to coco annotation files

# Author : Brian Isaac-Medina, brian.k.isaac-medina@durham.ac.uk
# Copyright (c) 2020 / 21 Department of Computer Science, Durham University, UK

# Dataset can be downloaded via https://www.kaggle.com/alejodosr/multirotor-aerial-vehicle-vid-mavvid-dataset

# Requires:
#   - tqdm

# Usage

# convert_mav_vid_to_coco.py [-h] [--root_dir ROOT_DIR] [--out_dir OUT_DIR]
#
#   optional arguments:
#     -h, --help           show this help message and exit
#     --root_dir ROOT_DIR  Root folder with the images and annotations
#     --out_dir OUT_DIR    Output directory to write the annotations file

#####################################################################
import argparse
import os
import json
import tqdm

parser = argparse.ArgumentParser(description="Converts MAV-VID yolo annotations to coco annotations")
parser.add_argument("--root_dir", type=str, help="Root folder with the images and annotations")
parser.add_argument("--out_dir", type=str, help="Output directory to write the annotations file")
opts = parser.parse_args()

root_dir = opts.root_dir
out_dir = opts.out_dir


dataset = {
    "info": {
        "year": 2020,
        "version": 1,
        "description": "Multirotor Aerial Vehicle VID (MAV-VID) dataset",
        "contributor": "alejodosr (Kaggle profile)",
        "url": "https://www.kaggle.com/alejodosr/multirotor-aerial-vehicle-vid-mavvid-dataset",
        "date_created": "2021-01-09"
    },
    "licenses": [{
        "id": 1,
        "name": "GPL 2",
        "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
    }],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "drone",
            "supercategory": "UAV"
        }
    ]
}

img_id = 1
ann_id = 1

for file in tqdm.tqdm([f for f in os.listdir(root_dir) if f.endswith(".txt")]):
    file_name = file[:-4]
    image = {
        "id": img_id,
        "file_name": file_name + ".jpg",
        "license": 1
    }
    with open(os.path.join(root_dir, file_name + ".shape"), 'r') as shape:
        line = shape.readline()
    H, W, _ = tuple(line.split(' '))
    H, W = int(H), int(W)
    image["width"] = W
    image["height"] = H
    dataset["images"].append(image)
    with open(os.path.join(root_dir, file), 'r') as objects_file:
        objects = objects_file.readlines()
    for obj in objects:
        _, x, y, w, h = tuple(obj.split(' '))
        x, y, w, h = float(x), float(y), float(w), float(h)
        x = (x - w / 2) * W
        y = (y - h / 2) * H
        w *= W
        h *= H
        annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": [],
            "area": w * h,
            "bbox": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
            "iscrowd": 0
        }
        dataset["annotations"].append(annotation)
        ann_id += 1
    img_id += 1
json.dump(dataset, open(os.path.join(out_dir, 'annotations.json'), 'w'))
