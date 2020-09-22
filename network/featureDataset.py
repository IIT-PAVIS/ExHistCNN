from __future__ import print_function, division
import os
import torch
import csv
import numpy as np

from skimage import io, transform
from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import Dataset, DataLoader
import utils.csv_io as c_io
from torchvision import transforms, utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

data_keys = ('depth', 'up','down','left','right')

class featureDataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding"""

    def __init__(self, csv_file, h, w, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gt_info = c_io.read_csv(csv_file)
        self.transform = transform
        self.height = h
        self.width = w

    def __len__(self):
        return len(self.gt_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        depth = io.imread(self.gt_info[idx][0]) # in mm

        resized_depth = transform.resize(depth, (self.height,  self.width))  # scaled between 0 to 1

        vismap_up = io.imread(self.gt_info[idx][1]) # 0,255

        resized_vismap_up = transform.resize(vismap_up, (self.height, self.width))  # scaled between 0 to 1
        vismap_down = io.imread(self.gt_info[idx][2])
        resized_vismap_down = transform.resize(vismap_down, (self.height, self.width))  # scaled between 0 to 1
        vismap_left = io.imread(self.gt_info[idx][3])
        resized_vismap_left = transform.resize(vismap_left, (self.height, self.width))  # scaled between 0 to 1
        vismap_right = io.imread(self.gt_info[idx][4])
        resized_vismap_right = transform.resize(vismap_right, (self.height, self.width))  # scaled between 0 to 1

        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][5])

        sample = {'depth': resized_depth, 'up': resized_vismap_up, 'down': resized_vismap_down, 'left': resized_vismap_left, 'right':resized_vismap_right, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


