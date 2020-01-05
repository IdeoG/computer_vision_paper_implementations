import torch.nn.functional as F
from torch import nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        """ ImageNet Classification with Deep Convolutional Neural Networks
            https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        """
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Sequential(nn.Linear(6 * 6 * 256, 4096), nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.Dropout(p=0.5))
        self.fc3 = nn.Linear(4096, 200)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = x.view(-1, 6 * 6 * 256)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output


if __name__ == '__main__':
    """
    ================================================================
    Total params: 59,100,744
    Trainable params: 59,100,744
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 5.09
    Params size (MB): 225.45
    Estimated Total Size (MB): 231.11
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 224, 224)
    summary(AlexNet(), input_size, device='cpu')
