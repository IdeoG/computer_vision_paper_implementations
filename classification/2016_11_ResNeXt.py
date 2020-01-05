from collections import OrderedDict

import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_relu=True, groups=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if use_relu else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.relu:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expantion = 4

    def __init__(self, inplanes, planes, downsample=None):
        super(Bottleneck, self).__init__()

        self.main_branch = nn.Sequential(OrderedDict([
            ('conv1', BasicConv(inplanes, planes, kernel_size=1)),
            ('conv2', BasicConv(planes, planes, kernel_size=3, padding=1)),
            ('conv3', BasicConv(planes, planes * self.expantion, kernel_size=1))
        ]))

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.main_branch(x)
        x = x + identity

        return self.relu(x)


class ResNeXt(nn.Module):
    """ Aggregated Residual Transformations for Deep Neural Networks
        https://arxiv.org/abs/1611.05431
    """

    def __init__(self, layers, n_classes=1000):
        super(ResNeXt, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.__make_block_layers(64, 64, layers[0])
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self.__make_block_layers(256, 128, layers[1])
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = self.__make_block_layers(512, 256, layers[2])
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = self.__make_block_layers(1024, 512, layers[3])
        self.avg_pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, n_classes)

    def __make_block_layers(self, inplanes, planes, n_blocks):
        downsample = BasicConv(inplanes, planes * Bottleneck.expantion, kernel_size=1, use_relu=False)

        return nn.Sequential(
            Bottleneck(inplanes, planes, downsample),
            *[Bottleneck(planes * Bottleneck.expantion, planes) for _ in range(n_blocks - 1)]
        )

    def forward(self, x):
        x = self.max_pool1(self.conv1(x))
        x = self.max_pool2(self.conv2(x))
        x = self.max_pool3(self.conv3(x))
        x = self.max_pool4(self.conv4(x))
        x = self.avg_pool5(self.conv5(x))
        x = x.view(-1, 2048)
        out = self.fc(x)
        return x


if __name__ == '__main__':
    """
    ================================================================
    Total params: 25,583,464
    Trainable params: 25,583,464
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 381.69
    Params size (MB): 97.59
    Estimated Total Size (MB): 479.85
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 224, 224)
    summary(ResNeXt(layers=[3, 4, 6, 3]), input_size, device='cpu')
