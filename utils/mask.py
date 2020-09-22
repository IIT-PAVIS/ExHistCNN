import open3d
import numpy as np
import math
import random
import cv2
from lib.reconstruction import Segment
from tools.commons.callbacks import cb_load_rgb_image
from lib.utils import improc
import os
from lib.utils.plot import imshow,pause
import pdb

"""
functions for computing rectangular mask of obj online
can be potentially merged with improc.py
"""   
    
def PCDtoImage(pcd,K,D,campose):
    """
    Project 3D points to image points
    Parameters
    ----------
    pcd : open3d pcd
    K: camera intrinsics matrix as numpy matrix 
    D: Distortion coefficients as numpy array
    campose: camera pose matrix as numpy matrix

    Returns
    -------
    imagePoints : image points as numpy matrix
    """   

    rvec, _ = cv2.Rodrigues(campose[:3,:3])
    
    tvec = campose[:3,3]
    xyz = np.asarray(pcd.points)
    
    imagePoints,_ = cv2.projectPoints(xyz, rvec, tvec, K, D)
    imagePoints = np.int32(imagePoints).reshape(-1,2)
    
    # within image range
    u = imagePoints[:,0]
    v = imagePoints[:,1]
    
    u [u > 479] = 479
    u [u < 0] = 0
    
    v [v > 639] = 639
    v [v < 0] = 0
    imagePoints[:,0] = u
    imagePoints[:,1] = v
    #print(imagePoints)
   
    return imagePoints

   
def cornersfromImagePoints(imagepoints):
    """
    get the bbx corner points from a list of image points  

    Parameters
    ----------
    imagepoints : numpy.ndarray of image points

    Returns
    -------
    contours : list of bbx corner points
    """   
    hull = cv2.convexHull(imagepoints)
    contour = hull
    return contour
    
def computeVisibilityFromMaskImage(segs, datascenepath,pid):
    """
    Prepare mask.
    
    Parameters
    ----------
    segs: list of Segments
    datascenepath: path to the folder holding masks
    D: index id of the camera pose
    
    Returns
    -------
    nums_vispixels_occ: N <seg_key, number of visible pixels accounting for occlusions> 
    nums_vispixels_senza_occ: N <seg_key, number of visible pixels without accounting for occlusions> 
    """ 
    nums_vispixels_occ = {}
    nums_vispixels_senza_occ = {}
   
    fname_occ = os.path.join(datascenepath, "mask","{:03d}".format(pid) + ".png")
    mask_occ = cb_load_rgb_image(fname_occ)
    
    mask_occ = improc.imresize(mask_occ, size=(640,480))
    
    mask_occ = mask_occ[:,:,0] - 1   

    for seg_key in segs: 
        gt_label = segs[seg_key].gt_label
        class_id = int(gt_label[4:])-1
        ## prepare the masks with occlusion for each view point (using gt_label)
        fg_temp = np.where(mask_occ == class_id)
        num_vispixels_occ = fg_temp[0].size
        nums_vispixels_occ[seg_key] = num_vispixels_occ 
       
        ## prepare the masks without occlusion for each obj (using gt_label)
        fname_senza_occ = os.path.join(datascenepath, "objmask","{:03d}".format(pid), str(class_id+1) + ".png")
        mask_senza_occ = cb_load_rgb_image(fname_senza_occ)
        mask_senza_occ = improc.imresize(mask_senza_occ, size=(640,480))
        mask_senza_occ = mask_senza_occ[:,:,0]
        fg_temp = np.where(mask_senza_occ>0)
        num_vispixels_senza_occ = fg_temp[0].size
        nums_vispixels_senza_occ[seg_key] = num_vispixels_senza_occ
   
    return nums_vispixels_occ, nums_vispixels_senza_occ

def computeVisibilityApprox(segs,K,D,campose, SceneReal):
    """
    Prepare mask.
    
    Parameters
    ----------
    segs: list of Segments
    K: camera intrinsics matrix as numpy matrix 
    D: Distortion coefficients as numpy array
    campose: camera pose matrix as numpy matrix
    
    Returns
    -------
    nums_vispixels_occ: N <seg_key, number of visible pixels accounting for occlusions> 
    nums_vispixels_senza_occ: N <seg_key, number of visible pixels without accounting for occlusions> 
    """ 
    # prepare the canvas
    mask = np.zeros([480,640],np.uint8)
    
    # sort the seg based on the distance to camera pose
    key_list = []
    distance_list = []
    cam_t = campose[:3,3]
    
    nums_vispixels_occ = {}
    nums_vispixels_senza_occ = {}
    
    for seg_key in segs:
        key_list.append(seg_key)
        seg = segs[seg_key]
        # compute the distance along camera z-axis
        unit_z_vect = campose[:3,:3] @ np.array([0,0,1])
        diff_t = np.array(seg.center)-cam_t
        distance_temp = diff_t @ unit_z_vect
        
        distance_list.append(distance_temp)
    
    # rank the distance in descending sequence 
    sorted_ind = sorted(range(len(distance_list)), key=distance_list.__getitem__) # ascending
    sorted_ind = np.flip(sorted_ind)  # descending (furthest -> nearest)
    
    # iterate from the furthest segmet, and overwrite the pixel value when updating each segmet (occlusion will be considered in this way)
    mask_value = 255 
    for index in  sorted_ind:
        seg_key = key_list[index]
        seg = segs[seg_key]
        # the pose should be the transfrom from the camera coordinate to the obj center
        if SceneReal:
            camera_pose_obj = np.linalg.inv(campose) @ seg.pose
        else:
            camera_pose_obj = np.linalg.inv(campose)
            
        raw_image_points = PCDtoImage(seg.cub_corner_model_pcd,K,D,camera_pose_obj)
        corner_points = cornersfromImagePoints(raw_image_points)
        
        obj_mask = np.zeros(mask.shape,np.uint8)
        cv2.drawContours(obj_mask,[corner_points],0,255,-1) # background = 0, mask = 255
        obj_points = np.transpose(np.nonzero(obj_mask))
        obj_size,_ = obj_points.shape
        nums_vispixels_senza_occ[seg_key] = obj_size
        mask[obj_points[:,0],obj_points[:,1]] = mask_value 
        mask_value  = mask_value - 10
        
    # update the visibility ratio (account for occlusion) in visibility 
    mask_value  = 255
    for index in  sorted_ind:
        seg_key = key_list[index]
        fg_temp = np.where(mask == mask_value)
        num_vispixels_occ = fg_temp[0].size
        nums_vispixels_occ[seg_key] = num_vispixels_occ
        mask_value = mask_value - 10
# mask shape is projected correctly (Validated on 28 March)    
#    imshow(mask)
#    pause()
    return nums_vispixels_occ, nums_vispixels_senza_occ

def obtainCroppedRGBDApprox(segs,K,D,campose,rgb,depth,SceneReal):
    """
    get the cropped depth and rgb image.
    
    Parameters
    ----------
    segs: list of Segments
    K: camera intrinsics matrix as numpy matrix 
    D: Distortion coefficients as numpy array
    campose: camera pose matrix as numpy matrix
    rgb
    depth
    Returns
    -------
    list of cropped rgb: N <seg_key, cropped rgb> 
    list of cropped depth: N <seg_key, cropped depth> 
    """ 

    
    # sort the seg based on the distance to camera pose
    
    rgb_crop_list = {}
    depth_crop_list = {}

    # iterate from the furthest segmet, and overwrite the pixel value when updating each segmet (occlusion will be considered in this way) 
    for seg_key in segs:
       
        seg = segs[seg_key]
        # the pose should be the transfrom from the camera coordinate to the obj center
        if SceneReal:
            camera_pose_obj = np.linalg.inv(campose) @ seg.pose
        else:
            camera_pose_obj = np.linalg.inv(campose)
        
        raw_image_points = PCDtoImage(seg.cub_corner_model_pcd,K,D,camera_pose_obj)
        corner_points = cornersfromImagePoints(raw_image_points)
        # get bounding box
        x,y,w,h = cv2.boundingRect(corner_points)
        margin = 5
        rmin, rmax, cmin, cmax = y ,y+h, x, x+w
        bbox = [rmin - margin, rmax + margin, cmin - margin, cmax + margin]
        rmin, rmax, cmin, cmax = bbox
        # clean data and crop input images
        rgb_crop = rgb[rmin:rmax, cmin:cmax]
        depth_crop = depth[rmin:rmax, cmin:cmax]
        rgb_crop_list[seg_key] = rgb_crop
        depth_crop_list[seg_key] = depth_crop
#        imshow(rgb_crop)
#        pause()
    
    return rgb_crop_list, depth_crop_list
        

def computeExploredVisibilityApprox(segs,K,D,campose):
    """
    
    Parameters
    ----------
    segs: list of Segments
    K: camera intrinsics matrix as numpy matrix 
    D: Distortion coefficients as numpy array
    campose: camera pose matrix as numpy matrix
    
    Returns
    -------
    nums_vispixels_explored: N <seg_key, number of visible pixels that have been seen before> 
    """ 
    # prepare the canvas
    mask = np.zeros([480,640],np.uint8)
    
    # sort the seg based on the distance to camera pose
    key_list = []
    distance_list = []
    cam_t = campose[:3,3]
    
    nums_vispixels_explored = {}
    
    for seg_key in segs:
        key_list.append(seg_key)
        seg = segs[seg_key]
        # compute the distance along camera z-axis
        unit_z_vect = campose[:3,:3] @ np.array([0,0,1])
        diff_t = np.array(seg.center)-cam_t
        distance_temp = diff_t @ unit_z_vect
        
        distance_list.append(distance_temp)
    
    # rank the distance in descending sequence 
    sorted_ind = sorted(range(len(distance_list)), key=distance_list.__getitem__) # ascending
    sorted_ind = np.flip(sorted_ind)  # descending (furthest -> nearest)
    
    # iterate from the furthest segmet, and overwrite the pixel value when updating each segmet (occlusion will be considered in this way)
    mask_value = 255 
    for index in  sorted_ind:
        seg_key = key_list[index]
        seg = segs[seg_key]
        # the pose should be the transfrom from the camera coordinate to the obj center
        camera_pose_obj = np.linalg.inv(campose) @ seg.pose 
        raw_image_points = PCDtoImage(seg.cub_corner_model_pcd,K,D,camera_pose_obj)
        corner_points = cornersfromImagePoints(raw_image_points)
        
        obj_mask = np.zeros(mask.shape,np.uint8)
        cv2.drawContours(obj_mask,[corner_points],0,255,-1) # background = 0, mask = 255
        obj_points = np.transpose(np.nonzero(obj_mask))
        obj_size,_ = obj_points.shape
        nums_vispixels_senza_occ[seg_key] = obj_size
        mask[obj_points[:,0],obj_points[:,1]] = mask_value 
        mask_value  = mask_value - 10
        
    # update the visibility ratio (account for occlusion) in visibility 
    mask_value  = 255
    for index in  sorted_ind:
        seg_key = key_list[index]
        fg_temp = np.where(mask == mask_value)
        num_vispixels_occ = fg_temp[0].size
        nums_vispixels_occ[seg_key] = num_vispixels_occ
        mask_value = mask_value - 10
## mask shape is projected correctly (Validated on 28 March)    
#    imshow(mask)
#    pause()
    return nums_vispixels_occ, nums_vispixels_senza_occ