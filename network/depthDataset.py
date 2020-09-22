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

class depthDataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding"""

    def __init__(self, csv_file, size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gt_info = c_io.read_csv(csv_file)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.gt_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        depth = io.imread(self.gt_info[idx][0]) # in mm
        resized_depth = transform.resize(depth, (self.size,  self.size))  # scaled between 0 to 1
        resized_depth = resized_depth.view().reshape((self.size,  self.size, 1))
        ### To edit! reshape this one to x*x*1
        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][-1])

        sample = {'sample': resized_depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

