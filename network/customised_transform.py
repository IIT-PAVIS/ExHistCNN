''' This file contains all customised transform functions
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020
'''

import torch
import numpy as np
from skimage import transform


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cuboid_data, label = sample['sample'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        cuboid_data = cuboid_data.transpose((2, 0, 1))
        return {'sample': torch.from_numpy(cuboid_data).float(),
                'label': torch.tensor(label)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        cuboid_data, label = sample['sample'], sample['label']

        h, w = cuboid_data.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        resized_data = transform.resize(cuboid_data, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'sample': resized_data, 'label': label}


class ToTensorF(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_keys = ('depth', 'up', 'down', 'left', 'right')
        for key in data_keys:
            img = sample[key]
            sample[key] = img.transpose((2, 0, 1))

        return {'depth':torch.from_numpy(sample['depth']).float(),
                'up': torch.from_numpy(sample['up']).float(),
                'down': torch.from_numpy(sample['down']).float(),
                'left':torch.from_numpy(sample['left']).float(),
                'right': torch.from_numpy(sample['right']).float(),
                'label': torch.tensor(sample['label'])}


class ToTensorFGeneric(object):
    """Convertone channel into three channels"""
    def __call__(self, sample):
        channels = len(sample)-1
        label = sample['label']
        sample_new = {}
        for i in range(channels):
            img = sample[str(i)]
            img_t = img.transpose((2, 0, 1))
            sample_new[str(i)] = torch.from_numpy(img_t).float()

        sample_new['label'] = label
        return sample_new


class ToThreeChannel(object):
    """Convertone channel into three channels"""
    def __call__(self, sample):
        data_keys = ('depth', 'up', 'down', 'left', 'right')
        for key in data_keys:
            img = sample[key]
            sample[key] = np.squeeze(np.stack((img,) * 3, -1))

        return {'depth':sample['depth'],
                'up':sample['up'],
                'down': sample['down'],
                'left': sample['left'],
                'right':sample['right'],
                'label': sample['label']}


class ToThreeChannelGeneric(object):
    """Convertone channel into three channels"""
    def __call__(self, sample):
        cuboiddata = sample['sample']
        channels = cuboiddata.shape[-1]

        label = sample['label']
        sample = {}
        for i in range(channels):
            img = np.squeeze(cuboiddata[:,:,i])
            img3 = np.squeeze(np.stack((img,) * 3, -1))
            sample[str(i)] = img3

        sample['label'] = label

        return sample


class ToTensor1D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        feature, label = sample['sample'], sample['label']
        return {'sample': torch.from_numpy(feature).float(),
                'label': torch.tensor(label)}