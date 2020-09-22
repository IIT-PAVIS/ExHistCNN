import torch
import torch.nn as nn
import torch.nn.functional as func


class depthCNN(nn.Module):
    def __init__(self, patchSize=64, useBatchNorm = False):
        super(depthCNN, self).__init__()
        self.useBatchNorm = useBatchNorm
        self.patchSize = patchSize
        # kernels
        # conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # 5 input image channel, 6 output channels, 3x3 square convolution,
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) ### stride = 1, the output dimension is not changed apart from the depth
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pool = nn.MaxPool2d(2, stride=2) # (Hin-2)/2+1, (Win-2)/2+1
        # apply drop out
        self.fc_dropout = nn.Dropout(p=0.5)    # https://arxiv.org/pdf/1207.0580.pdf
        # fully connected layers
        self.fc1 = nn.Linear(128 * int(patchSize/8) * int(patchSize/8), 120) ### divide by 8 because 3 maxpooling us applied
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4) # 4 output

        if useBatchNorm:
            self.conv1_bn = nn.BatchNorm2d(32)
            self.conv2_bn = nn.BatchNorm2d(64)
            self.conv3_bn = nn.BatchNorm2d(128)
            self.conv_dropout = nn.Dropout(p=0.2)   # http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf

    def forward(self, x):
        x = x.float()
        if self.useBatchNorm:
            x = self.pool(self.conv_dropout(func.relu(self.conv1_bn(self.conv1(x)))))
            x = self.pool(self.conv_dropout(func.relu(self.conv2_bn(self.conv2(x)))))
            x = self.pool(self.conv_dropout(func.relu(self.conv3_bn(self.conv3(x)))))
        else:
            x = self.pool(func.relu(self.conv1(x)))
            x = self.pool(func.relu(self.conv2(x)))
            x = self.pool(func.relu(self.conv3(x)))

        x = x.view(-1, 128 * int(self.patchSize/8) * int(self.patchSize/8))
        # x = self.dropout(x)
        x = self.fc_dropout(func.relu(self.fc1(x)))
        x = self.fc_dropout(func.relu(self.fc2(x)))
        x = self.fc3(x)
        return x