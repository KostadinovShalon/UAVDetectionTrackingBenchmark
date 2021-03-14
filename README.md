# UAVDetectionTrackingBenchmark

This repository contains the code, configuration files and dataset statistics used for the paper  
**Unmanned Aerial Vehicle Visual Detection and Tracking using Deep Neural Networks: A Performance Benchmark** submitted
to IROS 2021. 

The repository is organized as follows:

 - **datasets** (dir): Contains the COCO annotation files used for each dataset
 - **detection** (dir): This directory contains the configuration files for detection, the log files and some scripts
 used for setting up the detection dataset.

### Installation

Detection and tracking was carried using the OpenMMLab frameworks for each task. In this section, we give a summary
on how to setup the frameworks for each task

#### Detection

1. Install [MMDetection](https://github.com/open-mmlab/mmdetection) using the [Getting Started](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
guide.

1. Create a directory under the `configs` folder (e.g., `configs/uavbenchmark`) and copy the [config files](detection/configs).

1. Create the `data` directory in the root folder of the project and create the dataset folders (you can also use symbolic links):
    - anti-uav
    - anti-uav/images
    - drone-vs-bird
    - drone-vs-bird/images
    - mav-vid
    - mav-vid/images

1. Under each dataset folder copy the [annotation files](datasets) to its corresponding folder. 

1. Copy all the images for each dataset to `<dataset-folder>/images` (see below for each dataset details).

1. Create a `checkpoints` folder under the root of the project and download the weight files (see below) to this folder.

1. Run the following script: `python tools/test.py <PATH TO CONFIG FILE> <PATH TO WEIGHT FILE>`

### Datasets

Three datasets were in our benchmark. An example of each dataset is shown next, with (a) MAV-VID, (b) Drone-vs-Bird, 
(c) Anti-UAV Visual and (d) Anti-UAV Infrared. 

![Dataset examples](/images/dataset_example.png)

#### Multirotor Aerial Vehicle VID (MAV-VID)

This dataset consists on videos at different setups of single UAV. 
It contains videos captured from other drones, ground based surveillance cameras and handheld mobile devices.
It can be downloaded in its [kaggle site](https://www.kaggle.com/alejodosr/multirotor-aerial-vehicle-vid-mavvid-dataset). 


This dataset is composed of images with YOLO annotations divided in two directories: train and val. In order to
use this dataset in this benchmark kit, create the COCO annotation files for each data partition, using the 
[convert_mav_vid_to_coco.py](detection/utils/convert_mav_vid_to_coco.py), rename them to *train.json* and *val.json* and
move them to the `data/mav-vid` directory created in the installation steps. Then, copy all images of both partitions to 
the `data/mav-vid/images` directory.

#### Drone-vs-Bird
As part of the [International Workshop on Small-Drone Surveillance, Detection and Counteraction techniques](https://wosdetc2020.wordpress.com/drone-vs-bird-detection-challenge/)
of IEEE AVSS 2020, the main goal of this challenge is to reduce the high false positive rates that vision-based methods 
usually suffer. This dataset comprises videos of UAV captured at long distances and often surrounded by small objects, such as birds.

The videos can be downloaded upon request and the annotations can be downloaded via their [GitHub site](https://github.com/wosdetc/challenge).
The annotations follow a custom format, where a a .txt file is given for each video. Each annotation file has a line
for each video frame and the annotation is given in the format `<Frame number> <Number of Objects> <x> <y> <width> <height> [<x> <y> ...]`.
To use this dataset in this benchmark, first you need to convert the video to images via [video_to_images.py](detection/utils/video_to_images.py)
and then you need to create the COCO annotations using the [convert_drone_vs_bird_to_coco.py](detection/utils/convert_drone_vs_bird_to_coco.py) script.
Just as in the MAV-VID dataset, copy the images to the `data/drone-vs-bird/images` directory and the annotations to `data/drone-vs-bird`.

#### Anti-UAV
This multi-modal dataset comprises fully-annotated RGB and IR unaligned videos. Anti-UAV dataset is intended to provide 
a real-case benchmark for evaluating object tracker algorithms in the context of UAV. It contains recordings of 6 UAV 
models flying at different lightning and background conditions. This dataset can be downloaded in their [website](https://anti-uav.github.io/dataset/).

This dataset is also comprised of videos and custom annotations. Once downlaoded and extracted, the videos are organised 
in folders containing the RGB and IR versions, with their corresponding JSON annotations. To convert this dataset to 
images and COCO annotations, use the [convert_anti_uav_to_coco.py](detection/utils/convert_anti_uav_to_coco.py) script and copy
the images generated annotations to `data/anti-uav` and the images to `data/anti-uav/images`. The images folder will contain
the images for both modalities and the full (both modalities), RGB and IR annotations will be generated.

#### Dataset Statistics

Dataset object size

Dataset | Size | Average Object Size
--------|------|---------------------
**MAV-VID** | *Training*: 53 videos (29,500 images) <br /> *Validation*: 11 videos (10,732 images) | 136 x 77 pxs (0.66% of image size)
**Drone-vs-Bird** | *Training*: 61 videos (85,904 images) <br /> *Validation*: 16 videos (18,856 images) | 34 x 23 pxs (0.10% of image size)
**Anti-UAV** | *Training*: 60 videos (149,478 images) <br /> *Validation*: 140 videos (37,016 images) | *RGB*: 125 x 59 pxs (0.40% image size)<br />*IR*: 52 x 29 pxs (0.50% image size)

Location, size and image composition statistics

![Dataset examples](/images/dataset_statistics.png)

### Detection Results

Four detection architectures were used for our analysis: [Faster RCNN](https://arxiv.org/abs/1506.01497), [SSD512](https://arxiv.org/abs/1512.02325),
[YOLOv3](https://arxiv.org/abs/1804.02767) and [DETR](https://arxiv.org/abs/2005.12872). For the implementation details, refer
to our paper. The results are as follows:

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Model</th>
            <th>AP</th>
            <th>AP<sub>0.5</sub></th>
            <th>AP<sub>0.75</sub></th>
            <th>AP<sub>S</sub></th>
            <th>AP<sub>M</sub></th>
            <th>AP<sub>L</sub></th>
            <th>AR</th>
            <th>AR<sub>S</sub></th>
            <th>AR<sub>M</sub></th>
            <th>AR<sub>L</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>MAV-VID</td>
            <td>Faster RCNN <a href="detection/results/mav-vid/faster_rcnn_r50_fpn_2x_mav-vid_log.json">log</a> <a href="https://durhamuniversity-my.sharepoint.com/:u:/g/personal/pfvn47_durham_ac_uk/EauQ6qKxfjNChW0MxHqrB-4B1ai-MHW-tcE4FQOKoU41VA?e=WtqLfe">weights</a></td>
            <td>0.592</td>
            <td>0.978</td>
            <td>0.672</td>
            <td>0.154</td>
            <td>0.541</td>
            <td>0.656</td>
            <td>0.659</td>
            <td>0.369</td>
            <td>0.621</td>
            <td>0.721</td>
        </tr>
       <tr>
            <td>SSD512 <a href="detection/results/mav-vid/ssd512_vgg_2x_mav-vid_log.json">log</a> <a href="https://durhamuniversity-my.sharepoint.com/:u:/g/personal/pfvn47_durham_ac_uk/ESZvnNKS-i1Go2McAfiD4GQBKLV0N9Rg1J6rf6WLwdliyw?e=PfnowP">weights</a></td>
            <td>0.535</td>
            <td>0.967</td>
            <td>0.536</td>
            <td>0.083</td>
            <td>0.499</td>
            <td>0.587</td>
            <td>0.612</td>
            <td>0.377</td>
            <td>0.578</td>
            <td>0.666</td>
        </tr>
        <tr>
            <td>YOLOv3 <a href="detection/results/mav-vid/yolov3_d53_608_2x_mav-vid_log.json">log</a> <a href="https://durhamuniversity-my.sharepoint.com/:u:/g/personal/pfvn47_durham_ac_uk/EWFLnnBe-yxHvmT0eV9Hu0AB1PQOzCxF_VHz5qxmDIRqRw">weights</a></td>
            <td>0.537</td>
            <td>0.963</td>
            <td>0.542</td>
            <td>0.066</td>
            <td>0.471</td>
            <td>0.636</td>
            <td>0.612</td>
            <td>0.208</td>
            <td>0.559</td>
            <td>0.696</td>
        </tr>
        <tr>
            <td>DETR <a href="detection/results/mav-vid/detr_r50_8x2_2x_mav-vid_log.json">log</a> <a href="https://durhamuniversity-my.sharepoint.com/:u:/g/personal/pfvn47_durham_ac_uk/EdKV9tIeEZJMod54vFvdmKkBsYck0ofYaeBO-O-51_HL2w">weights</a></td>
            <td>0.545</td>
            <td>0.971</td>
            <td>0.560</td>
            <td>0.044</td>
            <td>0.490</td>
            <td>0.612</td>
            <td>0.692</td>
            <td>0.346</td>
            <td>0.661</td>
            <td>0.742</td>
        </tr>
    </tbody>
</table>


### Cite