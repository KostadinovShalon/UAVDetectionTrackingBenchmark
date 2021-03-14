#####################################################################

# Converts the Anti-UAV dataset to images and creates the coco annotation files

# Author : Brian Isaac-Medina, brian.k.isaac-medina@durham.ac.uk
# Copyright (c) 2020 / 21 Department of Computer Science, Durham University, UK

# Dataset can be downloaded via https://drive.google.com/open?id=1GICr5e9CZN0tcFM_VXhyogzxWD3LMvAw

# Requires:
#   - OpenCV
#   - tqdm

# Usage

#   convert_anti_uav_to_coco.py [-h] root_dir out_dir
#
#
#   positional arguments:
#     root_dir    Root folder with the dataset videos
#     out_dir     Output directory to write annotations and images
#
#   optional arguments:
#     -h, --help  show this help message and exit

#####################################################################

import argparse

import cv2
import os
import tqdm
import json
import copy

parser = argparse.ArgumentParser(description="Converts Anti-UAV videos to annotations")
parser.add_argument("root_dir", type=str, help="Root folder with the dataset videos")
parser.add_argument("out_dir", type=str, help="Output directory to write annotations and images")
opts = parser.parse_args()

root_dir = opts.root_dir
out_dir = opts.out_dir

allowed_videos = (".mp4", ".mpg", "avi")
os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

img_id = 1
ann_id = 1
create_images = False

video_dirs = [f for f in os.listdir(root_dir)]
total_videos = len(video_dirs)


def extract_images_and_annotations(video_dir_list, out_name_prefix):
    global img_id, ann_id

    dataset = {
        "info": {
            "year": 2020,
            "version": 1,
            "description": "Anti UAV dataset",
            "url": "https://anti-uav.github.io/dataset/",
            "date_created": "2021-02-08"
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

    dataset_IR = copy.deepcopy(dataset)
    dataset_RGB = copy.deepcopy(dataset)

    for vid_dir in tqdm.tqdm(video_dir_list):
        for vid in [f for f in os.listdir(os.path.join(root_dir, vid_dir)) if f.endswith(allowed_videos)]:
            vidcap = cv2.VideoCapture(os.path.join(root_dir, vid_dir, vid))

            annotations_file_path = f"{vid[:-4]}_label.json"
            annotations = json.load(open(os.path.join(root_dir, vid_dir, annotations_file_path)))
            annotations = annotations['gt_rect']

            is_ir = vid[:-4] == "IR"
            is_rgb = vid[:-4] == "RGB"

            hasFrames = True
            count = 0
            while hasFrames:
                hasFrames, image = vidcap.read()
                if hasFrames:
                    height, width = image.shape[:2]
                    file_name = f"{vid_dir}_{vid[:-4]}_{count}.jpg"
                    img = {
                        "id": img_id,
                        "file_name": file_name,
                        "license": 1,
                        "width": int(width),
                        "height": int(height)
                    }
                    dataset["images"].append(img)
                    if is_ir:
                        dataset_IR["images"].append(img)
                    if is_rgb:
                        dataset_RGB["images"].append(img)

                    bbox = annotations[count]
                    if len(bbox) == 4 and sum(bbox) > 0:
                        x, y, w, h = bbox
                        annotation = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "segmentation": [],
                            "area": w * h,
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "iscrowd": 0
                        }
                        dataset["annotations"].append(annotation)
                        if is_ir:
                            dataset_IR["annotations"].append(annotation)
                        if is_rgb:
                            dataset_RGB["annotations"].append(annotation)
                        ann_id += 1
                    if create_images:
                        cv2.imwrite(os.path.join(out_dir, "images/", file_name), image)
                    count += 1
                    img_id += 1
    json.dump(dataset, open(os.path.join(out_dir, f"{out_name_prefix}-full.json"), 'w'))
    json.dump(dataset_IR, open(os.path.join(out_dir, f"{out_name_prefix}-ir.json"), 'w'))
    json.dump(dataset_RGB, open(os.path.join(out_dir, f"{out_name_prefix}-rgb.json"), 'w'))


extract_images_and_annotations(video_dirs[:int(total_videos * 0.8)], "train")
extract_images_and_annotations(video_dirs[int(total_videos * 0.8):], "val")
