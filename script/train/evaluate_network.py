''' This script is to evaluate the trained models on test data
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020
'''

from __future__ import print_function
import torch
import torchvision.transforms as transforms
from network import cuboidCNN, depthDataset, depthCNN
from network import TwoStepH5Loader
from network import MLP, MLPDataset, ToTensor1D, ToTensor
from torch.utils.data import DataLoader
from sklearn import metrics
import time
import numpy as np
import os
import utils.filetool as ft
import utils.io as io
import glob
from os.path import expanduser

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)
number_workers = 1
batch_size = 1024
transform = transforms.Compose([ToTensor()])
classes = ("up", "down", "left", "right")

drop_last_batch = True

epoch_num = 149 # the epoch being selected
option = "MLPTwoStep2DScaled" #"CuboidTwoStep2DScaled" #"Depth" "MLP"
UseH5 = True
H5_list = ["CuboidTwoStep2D", "CuboidTwoStep4D", "CuboidTwoStep5D", "CuboidTwoStep2DScaled", "CuboidTwoStepUtilityOnly"]


# Function to evaluate the model with test data
# @Input: net, dataloader, device
# @Output: gt labels and predicted labels
def eval(net, dataloader, device):
    net.eval()
    with torch.no_grad():
        i = 0
        for sample_batched in dataloader:
            images, labels = sample_batched["sample"].to(device), sample_batched["label"].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            if i == 0:
                pred = predicted
                gt = labels
            else:
                pred = torch.cat((pred,predicted))
                gt = torch.cat((gt, labels))
            i = i+1

    return gt.cpu().numpy(), pred.cpu().numpy()


def load_data(folder):
    file_path = os.path.join(folder, "feature_all.npy")
    if not ft.file_exist(file_path):
        print("Save all data to {}".format(file_path))
        batch_files = glob.glob(os.path.join(folder, '*'))
        for i, file in enumerate(batch_files):

            print(file)
            data = np.load(file)
            if i == 0:
                concatenated = data
            else:
                concatenated = np.concatenate((concatenated, data))  # concatenate vertically
        np.save(file_path, concatenated)

    print("Load dataset from {}".format(file_path))
    dataset = MLPDataset(file_path, transform=ToTensor1D())
    return dataset


print("Loaded test data ... ")
if option in H5_list:
    h5_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'test', 'test.h5')
    csv_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'test', 'test.csv')
    data_loader_kwargs_h5 = {'option': option, 'h5_file': h5_file, 'csv_file': csv_file, 'transform': transform,
                             'batch_size': batch_size, 'shuffle': True, 'drop_last': drop_last_batch,
                             'num_workers': number_workers, }
    testloader = TwoStepH5Loader(**data_loader_kwargs_h5)  # batch size is usually set to 4, for debug, we can use 1

    if option == "CuboidTwoStep2D":
        net = cuboidCNN(64, True, inputChannel=2)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "CuboidTwoStep2DScaled":
        net = cuboidCNN(64, True, inputChannel=2)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "CuboidTwoStepUtilityOnly":
        net = cuboidCNN(64, True, inputChannel=1)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "CuboidTwoStep5D":
        net = cuboidCNN(64, True, inputChannel=5)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "CuboidTwoStep4D":
        net = cuboidCNN(64, True, inputChannel=4)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
else:
    if option == "MLP":
        test_folder = os.path.join(proj_path, "data", "featureset_cnn", "test")
        testset = load_data(test_folder)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers, drop_last=drop_last_batch)
        net = MLP(input_lenth=10240)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join("..", "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "MLPTwoStep2D":
        print("Evaluation for {}".format(option))
        test_folder = os.path.join(proj_path, "data", "featureset_cnn_twostep2D", "test")
        testset = load_data(test_folder)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers,
                                drop_last=drop_last_batch)
        net = MLP(input_lenth=4096)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "MLPTwoStep2DScaled":
        test_folder = os.path.join(proj_path, "data",  "featureset_cnn_twostep2DScaled", "test")
        testset = load_data(test_folder)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers,
                                drop_last=drop_last_batch)
        net = MLP(input_lenth=4096)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "MLPTwoStep5D":
        test_folder = os.path.join(proj_path, "data",  "featureset_cnn_twostep5D", "test")
        testset = load_data(test_folder)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers,
                                drop_last=drop_last_batch)
        net = MLP(input_lenth=10240)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "MLPTwoStep4D":
        test_folder = os.path.join(proj_path, "data", "featureset_cnn_twostep4D", "test")
        testset = load_data(test_folder)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers,
                                drop_last=drop_last_batch)
        net = MLP(input_lenth=8192)
        result_file = os.path.join(proj_path, "checkpoint", option, "testing_result_{:03d}.json".format(epoch_num))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint", option, "cp_{:03d}.pth".format(epoch_num)))
    elif option == "Depth":
        testset = depthDataset(csv_file=os.path.join(proj_path,"data","depth_cnn","test","test.csv"), size=64, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=number_workers, drop_last=drop_last_batch)
        net = depthCNN(64, True)
        result_file = os.path.join(os.path.join(proj_path,"checkpoint",option, "testing_result_{:03d}.json".format(epoch_num)))
        checkpoint = torch.load(os.path.join(proj_path, "checkpoint",option, "cp_{:03d}.pth".format(epoch_num)))

print("Initialized Network ... ")
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print(device)
net.to(device)
print("Loaded Network to GPU ... ")

# load already trained model
net.load_state_dict(checkpoint['model_state_dict'])
print("Loaded existing model check point ...")

start_t = time.time()

# evaluate
gt, pred = eval(net, testloader, device)

# compute the performance metrics
result = metrics.classification_report(gt, pred, digits=3)
print(result)

# save the loss and accuracy of train and validation into numpy array npz file.
io.write_json(result_file, result)

end_t = time.time()
print("Time for training  {:.03f} hrs.".format((end_t - start_t)/3600))
print('Finished Testing')




