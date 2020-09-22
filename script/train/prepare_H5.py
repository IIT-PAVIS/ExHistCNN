''' This script is to read/preprocess the dataset and save into H5 file to save time during the training and evaluation
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020

# The H5 file has mainly 4 fields:
 depth, vismap, scaled depth, and labels.
'''

from __future__ import print_function
import numpy as np
import os
import h5py
from skimage import io, transform
import utils.csv_io as c_io
from os.path import expanduser
import utils.improc as local_im

home = expanduser("~")
project_name = "auto3dmemo"
proj_path = os.path.join(home, project_name)

dataset_name = "mp3d"

class_list = ["up", "down", "left", "right"]
height = 64
width = 64


# main function to prepare the h5 file
def prepare_h5_from_csv(csv_file, output_h5_file):
    gt_info = c_io.read_csv(csv_file)
    num_items = len(gt_info)
    with h5py.File(output_h5_file, 'w') as f:
        depth_dset = None
        for idx in range(num_items):
            if depth_dset == None:
                depth_dset = f.create_dataset('depth', (num_items, height, width), dtype=np.float64)
                visibility_dset = f.create_dataset('vismap', (num_items, height, width), dtype=np.float64)
                depth_scale_dset = f.create_dataset('depth_scale', (num_items, height, width), dtype=np.float64)
                label_dset = f.create_dataset('labels', (num_items, 1), dtype=np.int8)

            depth = io.imread(gt_info[idx][0])  # in mm
            vismap_twostep = io.imread(gt_info[idx][1])  # 0,255

            resized_vismap_twostep = transform.resize(vismap_twostep, (height, width))  # scaled between 0 to 1

            half_large_fov = [0.94, 0.86]
            half_small_fov = [0.5, 0.39]
            scaled_depth = local_im.imshrink(depth, half_large_fov, half_small_fov, padding=True)
            resized_scaled_depth = transform.resize(scaled_depth, (height, width))  # scaled between 0 to 1
            resized_depth = transform.resize(depth, (height, width))  # scaled between 0 to 1

            depth_dset[idx] = resized_depth
            visibility_dset[idx] = resized_vismap_twostep
            depth_scale_dset[idx] = resized_scaled_depth
            label_dset[idx] = int(gt_info[idx][2])

            print('\tProcessed %d / %d images' % (idx, num_items))


print("For test set")
csv_file = os.path.join(proj_path,'data', dataset_name, 'dataset_cnn_twostep', 'test','test.csv')
output_h5_file = os.path.join(proj_path,'data', dataset_name, 'dataset_cnn_twostep', 'test','test.h5')
prepare_h5_from_csv(csv_file, output_h5_file)

print("For training set")
csv_file = os.path.join(proj_path, 'data', dataset_name, 'dataset_cnn_twostep', 'train', 'train.csv')
output_h5_file = os.path.join(proj_path,'data', dataset_name,  'dataset_cnn_twostep', 'train','train.h5')
prepare_h5_from_csv(csv_file, output_h5_file)

print("For validation set")
csv_file = os.path.join(proj_path,'data', dataset_name,  'dataset_cnn_twostep', 'validation','val.csv')
output_h5_file= os.path.join(proj_path,'data', dataset_name, 'dataset_cnn_twostep', 'validation','val.h5')
prepare_h5_from_csv(csv_file, output_h5_file)
print('Finished h5 files preparation!')




