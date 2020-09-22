''' This script is to train the ExHistCNN network
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020
'''

from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from network import cuboidCNN
from network import ToTensor
from network import TwoStepH5Loader
import torch.nn as nn
import time
import numpy as np
import os
from os.path import expanduser
import utils.filetool as ft

# defines the global parameters

# set the model to train
option = "UtilityOnly" #"2DScaled"  # "5D" "2DScaled"
batch_size = 1024
number_workers = 8
width = 64
height = 64
transform = transforms.Compose([ToTensor()])
drop_last_batch = True
criterion = nn.CrossEntropyLoss()
# the list of classes as output
classes = ("up","down","left","right")


# define which epoch to start with
start_epoch = 149
# define the total number of epoch to use
epoch_num = 200



# Function to load logged data
# @Input: the full path
# @Output: logged data
def load_train_result(path):
    result_data = np.load(path)
    acc_val=result_data["acc_val"]
    acc_train=result_data["acc_train"]
    loss_val = result_data["loss_val"]
    loss_train=result_data["loss_train"]

    return acc_val, acc_train, loss_val, loss_train


# Function to evaluate the model
# @Input: net, dataloader, device, the gt list of classes, flag for compute loss or not
# @Output: averaged accuracy and loss
def evaluate(net, dataloader, device, classes, loss_compute = True):
    net.eval()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    accuracy_cache = np.zeros(len(classes))
    running_loss = 0.0

    with torch.no_grad():
        for sample_batched in dataloader:
            images, labels = sample_batched["sample"].to(device), sample_batched["label"].to(device)
            outputs = net(images)
            if loss_compute:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    loss = running_loss/len(dataloader)
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        accuracy_cache[i] = class_correct[i] / class_total[i]

    return np.mean(accuracy_cache), loss


# Function to perform training in one epoch
# @Input: net, dataloader, optimizer, device
# @Output: loss at the epoch
def train_one_epoch(net, dataloader, optimizer, device):
    net.train()  # the only thing it does is setting the model to train mode
    running_loss = 0.0
    # train
    for i_batch, sample_batched in enumerate(dataloader):

        # get the inputs; data is a list of [inputs, labels]
        images_batch = sample_batched["sample"].to(device)
        labels_batch = sample_batched["label"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i_batch % print_freq == (print_freq - 1):  # print every 2000 mini-batches
            temp_loss = running_loss / print_freq
            print("\tIn mini-batch {:6d} ... ".format(i_batch))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, temp_loss))
            running_loss = 0.0

    return temp_loss


# Main script starts here
home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

if not ft.dir_exist(os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option))):
    os.mkdir(os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option)))

print("Loading training data ... ")
h5_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'train', 'train.h5')
csv_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'train', 'train.csv')
data_loader_kwargs_h5 = {'option': option, 'h5_file': h5_file, 'csv_file': csv_file, 'transform': transform, 'batch_size': batch_size, 'shuffle': True, 'drop_last': drop_last_batch, 'num_workers': number_workers,}
trainloader = TwoStepH5Loader(**data_loader_kwargs_h5)  # batch size is usually set to 4, for debug, we can use 1

print("Loading validation data ... ")
h5_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'validation', 'val.h5')
csv_file = os.path.join(proj_path, 'data', 'dataset_cnn_twostep', 'validation', 'val.csv')
data_loader_kwargs_h5 = {'option': option, 'h5_file': h5_file, 'csv_file': csv_file, 'transform': transform,
                         'batch_size': batch_size, 'shuffle': False, 'drop_last': drop_last_batch,
                         'num_workers': number_workers, }
valloader = TwoStepH5Loader(**data_loader_kwargs_h5)  # batch size is usually set to 4, for debug, we can use 1

print("set the network ...")
if "2D" in option:
    net = cuboidCNN(64, True, inputChannel=2)
elif "5D" in option:
    net = cuboidCNN(64, True, inputChannel=5)
elif "4D" in option:
    net = cuboidCNN(64, True, inputChannel=4)
elif option == "UtilityOnly":
    net = cuboidCNN(64, True, inputChannel=1)

print("Initialized Network ... ")
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print(device)
net.to(device)
print("Loaded Network to GPU ... ")

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# intialise the log data for accuracy and loss
acc_val = np.zeros(epoch_num)
acc_train = np.zeros(epoch_num)
loss_train = np.zeros(epoch_num)
loss_val = np.zeros(epoch_num)

# if having epoch saved, then load already trained model
if not start_epoch == 0:
    # load the existing model and resume training
    checkpoint = torch.load(os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option), "cp_{:03d}.pth".format(start_epoch)))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # load the existing training result
    acc_val_ex, acc_train_ex, loss_val_ex, loss_train_ex = load_train_result(os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option), "training_result_{:03d}.npz".format(epoch_num-1)))
    acc_val[:start_epoch] = acc_val_ex[:start_epoch]
    acc_train[:start_epoch] = acc_train_ex[:start_epoch]
    loss_val[:start_epoch] = loss_val_ex[:start_epoch]
    loss_train[:start_epoch] = loss_train_ex[:start_epoch]
    print("Loaded existing model check point ...")

# train the network
start_t = time.time()

print_freq = 20

for epoch in range(start_epoch, epoch_num,1):  # loop over the dataset multiple times
    print("In epoch {:6d} ... ".format(epoch))
    train_temp_loss = train_one_epoch(net,trainloader, optimizer, device)
    loss_train[epoch] = train_temp_loss

    print("\tPerform evaluation for current epoch ...")
    # evaluate now
    print("Validation set")
    acc_val[epoch],loss_val[epoch] = evaluate(net, valloader, device, classes)
    print("Training set")
    acc_train[epoch], _ = evaluate(net, trainloader, device, classes,loss_compute = False)

    # save epoch
    if epoch % 10 == 9:
        print("\tSave model for current epoch ...")
        path_checkpoint = os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option), 'cp_{:03d}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path_checkpoint)

    # save the loss and accuracy of train and validation into numpy array npz file.
    np.savez(os.path.join(proj_path, "checkpoint", "CuboidTwoStep{}".format(option), "training_result_{:03d}.npz".format(epoch_num - 1)), acc_val=acc_val, acc_train=acc_train, loss_val=loss_val, loss_train=loss_train)

end_t = time.time()

print("Time for training  {:.03f} hrs.".format((end_t - start_t)/3600))
print('Finished Training')




