#####################################################################

# Gets the Drone-vs-Bird images dataset to coco annotation files

# Author : Brian Isaac-Medina, brian.k.isaac-medina@durham.ac.uk
# Copyright (c) 2020 / 21 Department of Computer Science, Durham University, UK

# Dataset can be downloaded upon request. The dataset videos can be converted to images using the video_to_images.py
# script.

# Requires:
#   - PIL
#   - tqdm

# Usage

# convert_drone_vs_bird_to_coco.py [-h] [--root_dir ROOT_DIR] [--annotations_dir ANNOTATIONS_DIR] [--images_dir IMAGES_DIR] [--out_dir OUT_DIR]
#
# arguments:
#   -h, --help            show this help message and exit
#   --root_dir ROOT_DIR   Root folder with the dataset videos
#   --annotations_dir ANNOTATIONS_DIR
#                         Annotations directory
#   --images_dir IMAGES_DIR
#                         Images directory
#   --out_dir OUT_DIR     Output directory to write annotations and images


#####################################################################

import argparse
import os
import json
import tqdm
import random
from PIL import Image
import copy

parser = argparse.ArgumentParser(description="Converts Anti-UAV videos to annotations")
parser.add_argument("--root_dir", type=str, help="Root folder with the dataset videos")
parser.add_argument("--annotations_dir", type=str, help="Annotations directory")
parser.add_argument("--images_dir", type=str, help="Images directory")
parser.add_argument("--out_dir", type=str, help="Output directory to write annotations and images")
opts = parser.parse_args()

root_dir = opts.root_dir
annotations_dir = opts.annotations_dir
images_dir = opts.images_dir
out_dir = opts.out_dir


train_dataset = {
    "info": {
        "year": 2020,
        "version": 1,
        "description": "Drove-vs-bird 2020 Challenge dataset",
        "url": "https://wosdetc2020.wordpress.com/drone-vs-bird-detection-challenge/",
        "date_created": "2021-01-18"
    },
    "licenses": [{
        "id": 1,
        "name": "GPL 2",
        "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
    }],
    "images": [],
    "videos": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "drone",
            "supercategory": "UAV"
        }
    ]
}

test_dataset = {
    "info": {
        "year": 2020,
        "version": 1,
        "description": "Drove-vs-bird 2020 Challenge dataset",
        "url": "https://wosdetc2020.wordpress.com/drone-vs-bird-detection-challenge/",
        "date_created": "2021-01-18"
    },
    "licenses": [{
        "id": 1,
        "name": "GPL 2",
        "url": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
    }],
    "images": [],
    "videos": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "drone",
            "supercategory": "UAV"
        }
    ]
}

test_dataset = copy.deepcopy(train_dataset)

_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
total_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

# random.shuffle(_files)
# assume images are in order eg vid_0_0, vid_0_1, ... vid_0_-1, vid_1_0, vid_1_1, vid_1_-1, ...., vid_-1_0, vid_-1_1,...vid_-1_-1
allowed_videos = (".mp4", ".mpg", "avi")
vids = [f for f in os.listdir(root_dir) if f.endswith(allowed_videos)]
train_vid_last_idx = int(0.8*len(vids))
last_vid_idx_name = vids[train_vid_last_idx]
# train_last_idx = total_images.index(last_vid_idx_name)
train_last_idx = [id for id, m in enumerate(total_images) if last_vid_idx_name[:-4] in m][0]

# train_last_idx = int(0.8*len(_files))
train_files = _files[:train_last_idx]
test_files = _files[train_last_idx:]
video_id = 0
video_names = []

for dataset, files, output_name in ((train_dataset, train_files, "train.json"), (test_dataset, test_files, "val.json")):
    img_id = 1
    ann_id = 1
    for file in tqdm.tqdm(files):
        if file[:-4] not in video_names:
            video_names.append(file[:-4])
            dataset["videos"].append({"name": file[:-4], "id": video_id})
            video_id += 1
        with open(os.path.join(annotations_dir, file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            anns = line.split(' ')
            file_name = f"{file[:-4]}_{anns[0]}.jpg"
            if file_name not in total_images:
                break
            obs = int(anns[1])
            # if obs == 0:
            #     continue
            im = Image.open(os.path.join(images_dir, file_name))
            W, H = im.size
            vid_id = [id for id, _name in enumerate(video_names) if _name == file[:-4]][0]
            image = {
                "id": img_id,
                "file_name": file_name,
                "license": 1,
                "width": int(W),
                "height": int(H),
                "video_id": vid_id, # lookup id based off its name
                "frame_id": int(anns[0])
            }
            dataset["images"].append(image)
            for i in range(obs):
                x, y, w, h, clazs = anns[2 + i * 5], anns[3 + i * 5], anns[4 + i * 5], anns[5 + i * 5], anns[6 + i * 5]
                x, y, w, h = int(x), int(y), int(w), int(h)
                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [],
                    "area": w * h,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "instance_id": 0
                }
                dataset["annotations"].append(annotation)
                ann_id += 1
            img_id += 1

    json.dump(dataset, open(os.path.join(root_dir, output_name), 'w'))
