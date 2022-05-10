# Where to Explore Next? ExHistCNN for History-aware Autonomous 3D Exploration

Created by Yiming Wang and Alessio Del Bue, from the Pattern Analysis and Computer Vision (PAVIS) research line in Istituto Italiano di Tecnologia (IIT).

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=r_YE-oIccxQ
" target="_blank"><img src="http://img.youtube.com/vi/r_YE-oIccxQ/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a> </br>
  <a href="http://www.youtube.com/watch?v=r_YE-oIccxQ">Video presentation at ECCV 2020!</a>
</p>


## Introduction
In this project, we address the problem of autonomous 3D exploration of an unknown indoor environment using a depth camera. We cast the problem as the estimation of the Next Best View (NBV) that maximises the coverage of the unknown area. We do this by re-formulating NBV estimation as a classification problem and we propose a novel learning-based metric that encodes both, the current 3D observation (a depth frame) and the history of the ongoing reconstruction. One of the major contributions of this work is about introducing a new representation for the 3D reconstruction history as an auxiliary utility map which is efficiently coupled with the current depth observation. With both pieces of information, we train a light-weight CNN, named ExHistCNN, that estimates the NBV as a set of directions towards which the depth sensor finds most unexplored areas. 

In this repo, we will share the code, dataset as well our pretrained models. The shared scripts covers the dataset organisation and model training. While nbv-related scripts can not be fully shared due to redistribution constraints, we provide the full result data of the paper for potential comparison.


## Installation
The code is tested on Unbuntu 16.04 LTS with Cuda version V9.2.148. It is recommended to use to virtual environment, e.g. conda. You can create a conda virtual env using the provided environment file

`conda env create -f environment.yml`

Activate the virtual environment, you are good to go :)

## Dataset

Download the <a href="https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/yiming_wang_iit_it/EWReM9pGq1NOop3jxORE_S8BT9bkGg27uJUzLvdOHq6AfA?e=tbwQiw">dataset</a> through the link.
The dataset contains the rendered rooms in SUNCG and Matterport3D which are used in this paper together with their ground-truth annotation json files.
The renderings (depth+rgb) per room are generated with existing tools. For SUNCG, we use the [SUNCG toolbox](https://github.com/tinytangent/SUNCGtoolbox) while for real-world room scans from Matterport3D, we use [HabitatSim](https://github.com/facebookresearch/habitat-sim).
You can find the detailed procedure of dataset generation in the video:
<p align="center">
  <a href="https://www.youtube.com/watch?v=m1UtcLF0GpE" target="_blank"><img src="http://img.youtube.com/vi/m1UtcLF0GpE/0.jpg"
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a> </br>
  <a href="https://www.youtube.com/watch?v=m1UtcLF0GpE">Video for dataset generation!</a>
</p>

It is encouraged to put the dataset folder under the project, so that the script can be run without adapting paths.

### Dataset organisation
You can download the [csv files](https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/yiming_wang_iit_it/Ec1AvAnEpehLgOWY2E3ZEJUB3gkNDVAUcxaGJkNzJDxj-Q?e=SK9rde) that organise the dataset for train, validation and test.
It is recommended to put the dataset organisation files in the data folder.

Optionally, you can organise your own dataset split for training/validation/testing by tuning and running the scripts in *script/train/organise_dataset.py*.

### Dataset preprocessing
You should first pre-process the dataset, in terms of reading images and applying transforms using the script: *script/train/prepare_H5.py*.
The preprocessed data will be saved the data to h5 files. This facilitate speedy training.

In addition, if you want to train the MLP classifiers as described in the paper, you should prepare the resnet features for training by: *script/train/prepare_resnet_features_MLP.py*.
This should be done after you have the H5 files.

## Network
### Training
You can run *script/train/trainExHistCNN.py* to train the ExHistCNN models. You can also find the corresponding scripts to train the model with only depth images and MLP classifiers.

Optionally you can download the pre-trained models from [here](https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/yiming_wang_iit_it/EUBOjPb27VFOsNSuF7b8__EBoQ5WemMzOOxSJxdHyrnGAg?e=QKn4rz)
Please locate the pre-trained models within the checkpoint folder to avoid adapting path in the scripts.

### Evaluation
You can evaluate the models by running: *script/train/evaluate_network.py*. The evaluation metric will be saved.
You can make the performance plot by running: *script/train/plot_result.py*.

## Visualise NBV results
You can download the metadata for the [results](https://istitutoitalianotecnologia-my.sharepoint.com/:u:/g/personal/yiming_wang_iit_it/EQ9TTdvz-f1MpV2OEjpJ6ksBw_icmt5N6Uq9nUg41RUDKQ?e=wcUOIE) for each NBV startegies reported in the paper.
and run *script/result_analysis/analyse_result.py*, to obtain the figures as reported in the paper.

You can also visualise the point cloud at each time under each run and each method by running *script/result_analysis/reconstruct_NBV.py*.
## Issues
If you encounter issues in downloading the dataset or pre-trained models, please write to yiming.wang@iit.it, we will support you to download it.

## Citation
If you find our work useful in your research, please consider citing:
> @inproceedings{wang2020exhistcnn,\
    author = {Wang, Yiming and Del Bue, Alessio},\
    title = { Where to Explore Next? ExHistCNN for History-aware Autonomous 3D Exploration},\
    booktitle = {Proceedings of the European Conference on Computer Vision},\
    year = {2020}
}



