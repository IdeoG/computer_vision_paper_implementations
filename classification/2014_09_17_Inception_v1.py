import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels, block_channels):
        super(InceptionModule, self).__init__()

        self.conv1x1_1 = BasicConv(in_channels, block_channels[0], kernel_size=1)

        self.conv1x1_2 = BasicConv(in_channels, block_channels[1][0], kernel_size=1)
        self.conv3x3_2 = BasicConv(block_channels[1][0], block_channels[1][1], kernel_size=3, padding=1)

        self.conv1x1_3 = BasicConv(in_channels, block_channels[2][0], kernel_size=1)
        self.conv5x5_3 = BasicConv(block_channels[2][0], block_channels[2][1], kernel_size=5, padding=2)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = BasicConv(in_channels, block_channels[3], kernel_size=1)

    def forward(self, x):
        out_1 = self.conv1x1_1(x)

        out_2 = self.conv1x1_2(x)
        out_2 = self.conv3x3_2(out_2)

        out_3 = self.conv1x1_3(x)
        out_3 = self.conv5x5_3(out_3)

        out_4 = self.max_pool_4(x)
        out_4 = self.conv1x1_4(x)

        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)
        return out


class InceptionAuxiliary(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(InceptionAuxiliary, self).__init__()

        self.avg_pool = nn.AvgPool2d(5, 3)
        self.conv1x1 = BasicConv(in_channels, 128, 1)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = x.view(-1, 4 * 4 * 128)
        out = self.fc(x)
        return out


class InceptionV1(nn.Module):
    """ Going deeper with convolutions
        https://arxiv.org/pdf/1409.4842.pdf
    """

    def __init__(self, n_classes=1000):
        super(InceptionV1, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            BasicConv(64, 192, kernel_size=1),
            BasicConv(192, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block_3a = InceptionModule(192, [64, [96, 128], [16, 32], 32])
        self.block_3b = InceptionModule(256, [128, [128, 192], [32, 96], 64])

        self.max_pool_4 = nn.MaxPool2d(3, 2, padding=1)
        self.block_4a = InceptionModule(480, [192, [96, 208], [16, 48], 64])
        self.block_4b = InceptionModule(512, [160, [112, 224], [24, 64], 64])
        self.block_4c = InceptionModule(512, [128, [128, 256], [24, 64], 64])
        self.block_4d = InceptionModule(512, [112, [144, 288], [32, 64], 64])
        self.block_4e = InceptionModule(528, [256, [160, 320], [32, 128], 128])

        self.max_pool_5 = nn.MaxPool2d(3, 2, padding=1)
        self.block_5a = InceptionModule(832, [256, [160, 320], [32, 128], 128])
        self.block_5b = InceptionModule(832, [384, [192, 384], [48, 128], 128])

        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, n_classes)

        if self.training:
            self.aux_1 = InceptionAuxiliary(512, n_classes)
            self.aux_2 = InceptionAuxiliary(528, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.block_3a(x)
        x = self.block_3b(x)

        x = self.max_pool_4(x)
        x = self.block_4a(x)
        x = self.block_4b(x)
        x = self.block_4c(x)
        x = self.block_4d(x)
        x = self.block_4e(x)

        x = self.max_pool_5(x)
        x = self.block_5a(x)
        x = self.block_5b(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(-1, 1024)
        out = self.fc(x)
        return out


if __name__ == '__main__':
    """
    ================================================================
    Total params: 7,242,872
    Trainable params: 7,242,872
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 130.97
    Params size (MB): 27.63
    Estimated Total Size (MB): 159.17
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 224, 224)
    summary(InceptionV1(), input_size, device='cpu')
