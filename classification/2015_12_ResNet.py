import torch.nn as nn


class IdentityShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, projection=False):
        super(IdentityShortcut, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1) if projection else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.conv1x1:
            identity = self.conv1x1(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        return self.relu2(x)


class ResNet34(nn.Module):
    """ Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    """

    def __init__(self, n_classes=1000):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            IdentityShortcut(64, 64),
            IdentityShortcut(64, 64),
            IdentityShortcut(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            IdentityShortcut(64, 128, projection=True),
            IdentityShortcut(128, 128),
            IdentityShortcut(128, 128),
            IdentityShortcut(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            IdentityShortcut(128, 256, projection=True),
            IdentityShortcut(256, 256),
            IdentityShortcut(256, 256),
            IdentityShortcut(256, 256),
            IdentityShortcut(256, 256),
            IdentityShortcut(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv5 = nn.Sequential(
            IdentityShortcut(256, 512, projection=True),
            IdentityShortcut(512, 512),
            IdentityShortcut(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 512)
        out = self.fc(x)
        return out


if __name__ == '__main__':
    """
    ================================================================
    Total params: 21,804,264
    Trainable params: 21,804,264
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 83.37
    Params size (MB): 83.18
    Estimated Total Size (MB): 167.12
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 224, 224)
    summary(ResNet34(), input_size, device='cpu')
