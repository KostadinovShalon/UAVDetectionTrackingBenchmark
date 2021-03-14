#####################################################################

# Converts a video dataset to image dataset

# Author : Brian Isaac-Medina, brian.k.isaac-medina@durham.ac.uk
# Copyright (c) 2020 / 21 Department of Computer Science, Durham University, UK

# Requires:
#   - OpenCV
#   - tqdm

# Usage

# video_to_images.py [-h] root_dir out_dir
#
#   positional arguments:
#     root_dir    Videos dir
#     out_dir     Output images dir
#
#   optional arguments:
#     -h, --help  show this help message and exit

#####################################################################
import argparse

import cv2
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("root_dir", help="Videos dir")
parser.add_argument("out_dir", help="Output images dir")
opts = parser.parse_args()

root_dir = opts.root_dir
out_dir = opts.out_dir

allowed_videos = (".mp4", ".mpg", "avi")

for file in tqdm.tqdm([f for f in os.listdir(root_dir) if f.endswith(allowed_videos)]):
    os.makedirs(os.path.join(root_dir, 'images'), exist_ok=True)
    vidcap = cv2.VideoCapture(os.path.join(root_dir, file))
    hasFrames = True
    count = 0
    while hasFrames:
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(f"{os.path.join(out_dir, file[:-4])}_{count}.jpg", image)  # save frame as JPG file
            count += 1
