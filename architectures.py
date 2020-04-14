import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TinyConvNet(nn.Module):
    def __init__(self, imWidth, imHeight, numColorChannels, outputSize):
        super().__init__()
        self.convSize = 3
        self.numConvs = 2

        self.convOutputSize =   (imWidth - self.convSize + 1) *\
                                (imHeight - self.convSize + 1) *\
                                 self.numConvs
        print("conv1 output size: " + str(self.convOutputSize))

        self.conv1 = nn.Conv2d(numColorChannels, self.numConvs, self.convSize)
        self.l1 = nn.Linear(self.convOutputSize, outputSize)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.convOutputSize)
        x = F.relu(x)
        # x = F.max_pool2d(x, (2,2))
        x = self.l1(x)
        return x#F.softmax(x, dim=2)