import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(*args, **kwargs)
        self.bn = nn.BatchNorm2d(args[1])
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
