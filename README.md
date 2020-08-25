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

In this repo, we will share the code, dataset as well our pretrained models. Details will follow shortly!

## Citation
If you find our work useful in your research, please consider citing:
> @inproceedings{wang2020exhistcnn,\
    author = {Wang, Yiming and Del Bue, Alessio},\
    title = { Where to Explore Next? ExHistCNN for History-aware Autonomous 3D Exploration},\
    booktitle = {Proceedings of the European Conference on Computer Vision},\
    year = {2020}
}



