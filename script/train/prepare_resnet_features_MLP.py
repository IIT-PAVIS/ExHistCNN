''' This script is to prepare feature vectors that will be used to train MLP classifier.
We use resnet101 for feature extractor
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020

This script should be run after the H5 files are prepared
'''

from torchvision import models
import torch.nn as nn
from network import TwoStepH5Loader, ToThreeChannelGeneric, ToTensorFGeneric
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import os
import numpy as np
from os.path import expanduser
import utils.filetool as ft

resnet101 = models.resnet101(pretrained=True)
resnet101_feature = nn.Sequential(*list(resnet101.children())[:-1])

home = expanduser("~")
project_name = "auto3dmemo"
proj_path = os.path.join(home, project_name)

print("Initialized Network ... ")
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print(device)
resnet101_feature.to(device)

resnet101_feature.eval()

# load dataset
batch_size = 128
transform = transforms.Compose([ToThreeChannelGeneric(), ToTensorFGeneric()])

types = ["train", "validation", "test"]
option = "4D"
csv_files = {"train": os.path.join(proj_path, 'data', 'dataset_cnn_twostep','train','train.csv'), "validation": os.path.join(proj_path, 'data','dataset_cnn_twostep','validation','val.csv'), "test": os.path.join(proj_path, 'data','dataset_cnn_twostep','test','test.csv')}
h5_files = {"train": os.path.join(proj_path, 'data', 'dataset_cnn_twostep','train','train.h5'), "validation": os.path.join(proj_path, 'data', 'dataset_cnn_twostep','validation','val.h5'), "test": os.path.join(proj_path, 'data', 'dataset_cnn_twostep','test','test.h5')}

for type in types:
    print("Loaded {} data ... ".format(type))
    drop_last_batch = False
    if type == "train":
        drop_last_batch = True
    h5_file = h5_files[type]
    csv_file = csv_files[type]
    data_loader_kwargs_h5 = {'option': option, 'h5_file': h5_file, 'csv_file': csv_file, 'transform': transform,
                             'batch_size': batch_size, 'shuffle': False, 'drop_last': drop_last_batch,
                             'num_workers': 8, }
    dataloader = TwoStepH5Loader(**data_loader_kwargs_h5)

    if "2D" in option:
        channels = 2
    elif "5D" in option:
        channels = 5
    elif "4D" in option:
        channels = 4

    if not ft.dir_exist(os.path.join(proj_path, 'data', 'featureset_cnn_twostep{}'.format(option), type)):
        os.makedirs(os.path.join(proj_path, 'data', 'featureset_cnn_twostep{}'.format(option), type))

    # to read image and save feature
    for i_batch, sample_batched in enumerate(dataloader):
        print("\tIn mini-batch {:6d} ... ".format(i_batch))

        file_to_save = os.path.join(proj_path, 'data', 'featureset_cnn_twostep{}'.format(option), type, "batch_{:04d}.npy".format(i_batch))
        if not os.path.exists(file_to_save):
            # get the inputs; data is a list of [inputs, labels]
            for i in range(0, channels):
                key_img = str(i)
                img_batch = sample_batched[key_img].to(device)
                feature_batch = resnet101_feature(img_batch)

                if i == 0:
                    feature_batch = feature_batch.cpu()
                    concatenated = feature_batch.view(feature_batch.size()[0],feature_batch.size()[1]) # should be in Tensor now
                    print("concatenated size as in a mini-batch", concatenated.size())
                else:
                    feature_batch = feature_batch.cpu()
                    concatenated = torch.cat((concatenated, feature_batch.view(feature_batch.size()[0],feature_batch.size()[1])), 1)
                    print("\tconcatenated size as in a mini-batch", concatenated.size())
            # make the last column the label

            batch_labels = sample_batched["label"].view(sample_batched["label"].size()[0], 1).float()
            print(batch_labels.size())
            concatenated = torch.cat((concatenated, batch_labels), 1)

            # save numpy
            batch_features_np = concatenated.detach().numpy()  ## back to numpy
            print("Save feature vectors for batch {:04d}".format(i_batch))
            np.save(file_to_save, batch_features_np)
        else:
            print("File exists, skip ...")


