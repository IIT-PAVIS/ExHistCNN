''' This script is to train the MLP classifier
Author: Yiming Wang (yiming.wang@iit.it)
Last updated: July 2020
'''


from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from network import MLP, MLPDataset,ToTensor1D
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import glob
from os.path import expanduser
import utils.filetool as ft

home = expanduser("~")
project_name = "ExHistCNN"
proj_path = os.path.join(home, project_name)

option = "4D"
batch_size = 1024
transform = transforms.Compose([ToTensor1D()])
criterion = nn.CrossEntropyLoss()
classes = ("up","down","left","right")

drop_last_batch = True

start_epoch = 0
epoch_num = 150

acc_val = np.zeros(epoch_num)
acc_train = np.zeros(epoch_num)
loss_train = np.zeros(epoch_num)
loss_val = np.zeros(epoch_num)


def load_train_result(path):
    result_data = np.load(path)
    acc_val=result_data["acc_val"]
    acc_train=result_data["acc_train"]
    loss_val = result_data["loss_val"]
    loss_train=result_data["loss_train"]
    return acc_val, acc_train, loss_val, loss_train


def evaluate(net, dataloader, device, classes,loss_compute = True):
    net.eval()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    accuracy_cache = np.zeros(len(classes))
    running_loss = 0.0

    with torch.no_grad():
        for sample_batched in dataloader:
            features, labels = sample_batched["sample"].to(device), sample_batched["label"].to(device)

            outputs = net(features)
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
        if i_batch % print_freq == (print_freq - 1):
            temp_loss = running_loss / print_freq
            print("\tIn mini-batch {:6d} ... ".format(i_batch))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, temp_loss))
            running_loss = 0.0

    return temp_loss


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

print("Load validation data ... ")
val_folder = os.path.join(proj_path, 'data', 'featureset_cnn_twostep{}'.format(option), 'validation')
valset = load_data(val_folder)
valloader = DataLoader(valset, batch_size = batch_size, shuffle=False, num_workers= 8, drop_last = drop_last_batch )


print("Load training data ... ")
train_folder = os.path.join(proj_path, 'data', 'featureset_cnn_twostep{}'.format(option), 'train')
trainset = load_data(train_folder)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers=8, drop_last = drop_last_batch ) # batch size is usually set to 4, for debug, we can use 1

print("Initialize Network ... ")
if '2D' in option:
    net = MLP(input_lenth=4096)
elif '5D' in option:
    net = MLP(input_lenth=10240)
elif '4D' in option:
    net = MLP(input_lenth= 8192)
train_GPU = True
device = torch.device("cuda" if (torch.cuda.is_available() and train_GPU) else "cpu")
print(device)
net.to(device)
print("Loaded Network to GPU ... ")

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if not ft.dir_exist(os.path.join(proj_path, "checkpoint","MLPTwoStep{}".format(option))):
    os.makedirs(os.path.join(proj_path, "checkpoint","MLPTwoStep{}".format(option)))

# if having epoch saved, then load already trained model
if not start_epoch == 0:
    # load the existing model and resume training
    checkpoint = torch.load(os.path.join(proj_path, "checkpoint","MLPTwoStep{}".format(option),"cp_{:03d}.pth".format(start_epoch)))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # load the existing training result
    acc_val_ex, acc_train_ex, loss_val_ex, loss_train_ex = load_train_result(os.path.join(proj_path, "checkpoint", "MLPTwoStep{}".format(option), "training_result_{:03d}.npz".format(start_epoch)))
    acc_val[:start_epoch] = acc_val_ex[:start_epoch]
    acc_train[:start_epoch] = acc_train_ex[:start_epoch]
    loss_val[:start_epoch] = loss_val_ex[:start_epoch]
    loss_train[:start_epoch] = loss_train_ex[:start_epoch]
    print("Loaded existing model check point ...")

# train the network
start_t = time.time()

print_freq = 20

for epoch in range(start_epoch, epoch_num, 1):  # loop over the dataset multiple times
    print("In epoch {:6d} ... ".format(epoch))
    train_temp_loss = train_one_epoch(net, trainloader, optimizer, device)
    loss_train[epoch] = train_temp_loss

    # evaluate now
    acc_val[epoch], loss_val[epoch] = evaluate(net, valloader, device, classes)
    acc_train[epoch], _ = evaluate(net, trainloader, device, classes,loss_compute = False)

    # save epoch
    if epoch % 10 == 9:
        path_checkpoint = os.path.join(proj_path, "checkpoint", "MLPTwoStep{}".format(option), 'cp_{:03d}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path_checkpoint)

    # save the loss and accuracy of train and validation into numpy array npz file.
    np.savez(os.path.join(proj_path, "checkpoint", "MLPTwoStep{}".format(option), "training_result_{:03d}.npz".format(epoch_num - 1)), acc_val=acc_val, acc_train=acc_train, loss_val=loss_val, loss_train=loss_train)

end_t = time.time()

print("Time for training  {:.03f} hrs.".format((end_t - start_t)/3600))
print('Finished Training')




