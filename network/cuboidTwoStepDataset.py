from __future__ import print_function, division
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import utils.csv_io as c_io
import h5py
from utils.improc import get_tri_quarter,imshrink
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class cuboidTwoStepUtilityOnlyDataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding with two step depth for GT label"""

    def __init__(self, csv_file, h, w, transform = None):
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
        vismap_twostep = io.imread(self.gt_info[idx][1]) # 0,255
        resized_vismap_twostep = transform.resize(vismap_twostep, (self.height, self.width))  # scaled between 0 to 1

        # make cuboid data here
        cuboiddata = resized_vismap_twostep.view().reshape((self.height, self.width, 1))
        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][2])

        sample = {'sample': cuboiddata, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

class cuboidTwoStepUtilityOnlyDatasetH5(Dataset):
    def __init__(self, data_h5, gt_info, transform = None):
        self.transform = transform
        self.data_h5 = data_h5
        self.gt_info = gt_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.data_h5, 'r') as f:
            vismap = np.asarray(f['vismap'][idx], dtype = np.float64)
            label = int(self.gt_info[idx][2])

        cuboiddata = vismap.view().reshape((vismap.shape[0], vismap.shape[1], 1))

        sample = {'sample': cuboiddata, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.gt_info) # len(self.all_labels)

    def classes(self):
        return len(self.gt_info)

class cuboidTwoStep5DatasetH5(Dataset):
    def __init__(self, data_h5, gt_info, transform = None):
        self.transform = transform
        self.data_h5 = data_h5
        self.label_list = ["up", "down", "left", "right"]
        self.gt_info = gt_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.data_h5, 'r') as f:
            depth = np.asarray(f['depth'][idx], dtype = np.float64)
            vismap = np.asarray(f['vismap'][idx], dtype = np.float64)

        cuboiddata = depth
        label_id = int(self.gt_info[idx][2])

        # divide and compute utility map with
        for j, label in enumerate(self.label_list):
            resized_vismap_direction_temp = get_tri_quarter(vismap, label)
            cuboiddata = np.dstack((cuboiddata, resized_vismap_direction_temp))

        sample = {'sample': cuboiddata, 'label': int(label_id)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.gt_info)# len(self.all_labels)

    def classes(self):
        return len(self.gt_info)


class cuboidTwoStep4DatasetH5(Dataset):
    def __init__(self, data_h5, gt_info, transform = None):
        self.transform = transform
        self.data_h5 = data_h5
        self.label_list = ["up", "down", "left", "right"]
        self.gt_info = gt_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.data_h5, 'r') as f:
            vismap = np.asarray(f['vismap'][idx], dtype = np.float64)

        label_id = int(self.gt_info[idx][2])

        # divide and compute utility map with
        for j, label in enumerate(self.label_list):
            resized_vismap_direction_temp = get_tri_quarter(vismap, label)
            if j == 0:
                cuboiddata = resized_vismap_direction_temp
            else:
                cuboiddata = np.dstack((cuboiddata, resized_vismap_direction_temp))

        sample = {'sample': cuboiddata, 'label': int(label_id)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.gt_info)# len(self.all_labels)

    def classes(self):
        return len(self.gt_info)

class cuboidTwoStep5Dataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding"""

    def __init__(self, csv_file, h, w, transform = None):
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
        self.label_list = ["up", "down", "left", "right"]

    def __len__(self):
        return len(self.gt_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        depth = io.imread(self.gt_info[idx][0]) # in mm
        resized_depth = transform.resize(depth, (self.height,  self.width))  # scaled between 0 to 1
        cuboiddata = resized_depth
        vismap_twostep = io.imread(self.gt_info[idx][1])  # 0,255
        resized_vismap_twostep = transform.resize(vismap_twostep, (self.height, self.width))  # scaled between 0 to 1

        # divide and compute utility map with
        for j, label in enumerate(self.label_list):
            resized_vismap_direction_temp = get_tri_quarter(resized_vismap_twostep, label)
            cuboiddata = np.dstack((cuboiddata,resized_vismap_direction_temp))

        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][2])

        sample = {'sample': cuboiddata, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class cuboidTwoStep2DatasetH5(Dataset):
    def __init__(self, data_h5, gt_info, transform = None):
        self.transform = transform
        self.data_h5 = data_h5
        self.gt_info = gt_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with h5py.File(self.data_h5, 'r') as f:
            depth = np.asarray(f['depth'][idx], dtype = np.float64)
            vismap = np.asarray(f['vismap'][idx], dtype = np.float64)

        label = int(self.gt_info[idx][2])
        cuboiddata = np.dstack((depth, vismap))

        sample = {'sample': cuboiddata, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.gt_info)# len(self.all_labels)

    def classes(self):
        return len(self.gt_info)


class TwoStepH5Loader(DataLoader):
    def __init__(self, **kwargs):
        h5_file = kwargs.pop('h5_file')
        csv_file = kwargs.pop('csv_file')
        transform_ = kwargs.pop('transform')
        option = kwargs.pop('option')
        self.data_h5 = h5_file
        self.gt_info = c_io.read_csv(csv_file)

        if option == "2DScaled" or option == "CuboidTwoStep2DScaled":
            print("Reading data from {}".format(h5_file))
            self.dataset = cuboidTwoStep2DScaledDatasetH5(self.data_h5, self.gt_info, transform_)
        elif option == "2D" or option == "CuboidTwoStep2D":
            self.dataset = cuboidTwoStep2DatasetH5(self.data_h5, self.gt_info, transform_)
        elif option == "5D" or option == "CuboidTwoStep5D":
            self.dataset = cuboidTwoStep5DatasetH5(self.data_h5, self.gt_info, transform_)
        elif option == "4D" or option == "CuboidTwoStep4D":
            self.dataset = cuboidTwoStep4DatasetH5(self.data_h5, self.gt_info, transform_)
        elif option == "UtilityOnly" or option == "CuboidTwoStepUtilityOnly":
            self.dataset = cuboidTwoStepUtilityOnlyDatasetH5(self.data_h5, self.gt_info, transform_)
        super(TwoStepH5Loader, self).__init__(self.dataset, **kwargs)

    def close(self):
        if self.data_h5 is not None:
            self.data_h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def classes(self):
        return self.dataset.classes()


class cuboidTwoStep2DScaledDatasetH5(Dataset):
    def __init__(self, data_h5, gt_info, transform = None):
        self.transform = transform
        self.data_h5 = data_h5
        self.gt_info = gt_info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with h5py.File(self.data_h5, 'r') as f:
            depth_scaled = np.asarray(f['depth_scale'][idx], dtype = np.float64)
            vismap = np.asarray(f['vismap'][idx], dtype = np.float64)
        label = int(self.gt_info[idx][2])
        cuboiddata = np.dstack((depth_scaled, vismap))

        # print("cuboiddata shape", cuboiddata.shape)
        # print("cuboiddata type", type(cuboiddata[0,0,0]))
        # print("cuboiddata values", cuboiddata[32])
        #
        # print("label type", type(label))
        # print("label values", label)

        sample = {'sample': cuboiddata, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.gt_info) # len(self.all_labels)

    def classes(self):
        return len(self.gt_info)


class cuboidTwoStep2DScaledDataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding with two step depth for GT label"""

    def __init__(self, csv_file, h, w, transform = None):
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
        half_large_fov = [0.94, 0.86]
        half_small_fov = [0.5, 0.39]
        scaled_depth = imshrink(depth, half_large_fov, half_small_fov, padding=True)

        resized_depth = transform.resize(scaled_depth, (self.height,  self.width))  # scaled between 0 to 1

        vismap_twostep = io.imread(self.gt_info[idx][1]) # 0,255
        resized_vismap_twostep = transform.resize(vismap_twostep, (self.height, self.width))  # scaled between 0 to 1

        # make cuboid data here
        cuboiddata = np.dstack((resized_depth,resized_vismap_twostep))

        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][2])

        # print("cuboiddata shape", cuboiddata.shape)
        # print("cuboiddata type", type(cuboiddata[0, 0, 0]))
        # print("cuboiddata values", cuboiddata[32])
        #
        # print("label type", type(label))
        # print("label values", label)

        sample = {'sample': cuboiddata, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class cuboidTwoStep2Dataset(Dataset):
    """cuboid CNN Dataset for auto3d with memo encoding with two step depth for GT label"""

    def __init__(self, csv_file, h, w, transform = None):
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

        vismap_twostep = io.imread(self.gt_info[idx][1]) # 0,255
        resized_vismap_twostep = transform.resize(vismap_twostep, (self.height, self.width))  # scaled between 0 to 1

        # make cuboid data here
        cuboiddata = np.dstack((resized_depth,resized_vismap_twostep))

        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.gt_info[idx][2])

        sample = {'sample': cuboiddata, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


