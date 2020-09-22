import torch
import torch.nn as nn
import torch.nn.functional as func


class MLP(nn.Module):
    def __init__(self, input_lenth = 10240):
        super(MLP, self).__init__()
        self.input_lenth = input_lenth
        # apply drop out
        self.fc_dropout = nn.Dropout(p=0.5)    # https://arxiv.org/pdf/1207.0580.pdf
        # fully connected layers
        self.fc1 = nn.Linear(input_lenth, 1024) ### divide by 8 because 3 maxpooling us applied
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 4)  # 4 output

    def forward(self, x):
        x = x.float()
        x = x.view(-1, self.input_lenth)
        x = self.fc_dropout(func.relu(self.fc1(x)))
        x = self.fc_dropout(func.relu(self.fc2(x)))
        x = self.fc_dropout(func.relu(self.fc3(x)))
        x = self.fc4(x)
        return x