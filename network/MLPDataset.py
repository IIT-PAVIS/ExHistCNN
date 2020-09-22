from __future__ import print_function, division

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MLPDataset(Dataset):
    """MLP feature vector dataset for auto3d with memo encoding"""

    def __init__(self, file_path, transform=None):
        scaler = StandardScaler(copy=False)  # to normalise data
        all_data = np.load(file_path)
        print('all data shape',all_data.shape)
        feature_vecs = all_data[:, :-1]
        print('feature data shape', feature_vecs.shape)
        print("Making normalisation ... ")
        scaler.fit(feature_vecs)
        self.features = scaler.transform(feature_vecs)
        self.labels = all_data[:, -1]
        print('label data shape', self.labels.shape)
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # make cuboid data here
        feature = self.features[idx, :]

        # https://scikit-image.org/docs/dev/api/skimage.transform resize
        label = int(self.labels[idx].item())
        sample = {'sample': feature, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


