from collections import OrderedDict

import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3)), nn.ReLU())),
            ('conv1_2', nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3)), nn.ReLU())),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.layer2 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3)), nn.ReLU())),
            ('conv2_2', nn.Sequential(nn.Conv2d(128, 128, kernel_size=(3, 3)), nn.ReLU())),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.layer3 = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3)), nn.ReLU())),
            ('conv3_2', nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3)), nn.ReLU())),
            ('conv3_3', nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3)), nn.ReLU())),
            ('pool3', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.layer4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3)), nn.ReLU())),
            ('conv4_2', nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3)), nn.ReLU())),
            ('conv4_3', nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3)), nn.ReLU())),
            ('pool4', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(nn.Linear(512 * 8 * 8, 4096), nn.ReLU(), nn.Dropout())),
            ('fc2', nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout())),
            ('fc3', nn.Linear(4096, n_classes)),
        ]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, 512 * 8 * 8)

        return self.fc(x)


if __name__ == '__main__':
    """
    ================================================================
    Total params: 162,735,400
    Trainable params: 162,735,400
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 186.94
    Params size (MB): 620.79
    Estimated Total Size (MB): 808.30
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 224, 224)
    summary(VGG16(), input_size, device='cpu')
