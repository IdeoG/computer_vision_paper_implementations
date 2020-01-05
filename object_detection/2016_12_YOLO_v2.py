import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(*args, **kwargs)
        self.bn = nn.BatchNorm2d(args[1])
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class DarknetYoloV2(nn.Module):
    def __init__(self, n_classes=1000):
        super(DarknetYoloV2, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(3, 32, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )  # 1 layers
        self.conv2 = nn.Sequential(
            BasicConv(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )  # 2 layers
        self.conv3 = nn.Sequential(
            BasicConv(64, 128, 3, padding=1),
            BasicConv(128, 64, 1),
            BasicConv(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )  # 5 layers
        self.conv4 = nn.Sequential(
            BasicConv(128, 256, 3, padding=1),
            BasicConv(256, 128, 1),
            BasicConv(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )  # 8 layers
        self.conv5 = nn.Sequential(
            BasicConv(256, 512, 3, padding=1),
            BasicConv(512, 256, 1),
            BasicConv(256, 512, 3, padding=1),
            BasicConv(512, 256, 1),
            BasicConv(256, 512, 3, padding=1)
        )  # 13 layers
        self.conv6 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            BasicConv(512, 1024, 3, padding=1),
            BasicConv(1024, 512, 1),
            BasicConv(512, 1024, 3, padding=1),
            BasicConv(1024, 512, 1),
            BasicConv(512, 1024, 3, padding=1)
        )  # 18 layers

        self.fc = nn.Sequential(
            BasicConv(1024, n_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )  # 19 layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return self.fc(x)


class YoloV2(nn.Module):
    """ YOLO9000: Better, Faster, Stronger
        https://arxiv.org/abs/1612.08242
    """
    __priors = [
        (1.3221, 1.73145),
        (3.19275, 4.00944),
        (5.05587, 8.09892),
        (9.47112, 4.84053),
        (11.2364, 10.0071)
    ]

    def __init__(self, darknet, n_classes=20):
        super(YoloV2, self).__init__()

        self.S = 13
        self.n_classes = n_classes
        self.n_boxes = 5
        self.priors = self.__priors

        self.yolo_head = self.__load_yolo_head(darknet)  # 18 layers trained on ImageNet
        self.conv7 = nn.Sequential(
            BasicConv(1024, 1024, 3, padding=1),
            BasicConv(1024, 1024, 3, padding=1),
            BasicConv(1024, 1024, 3, padding=1)
        )

        self.conv8 = BasicConv(2048 + 1024, self.n_boxes * (1 + 4 + self.n_classes), 1)

    def forward(self, x):
        x = self.yolo_head.conv1(x)
        x = self.yolo_head.conv2(x)
        x = self.yolo_head.conv3(x)
        x = self.yolo_head.conv4(x)
        x = self.yolo_head.conv5(x)
        passthrough = x.view(x.size(0), -1, self.S, self.S)

        x = self.yolo_head.conv6(x)
        x = self.conv7(x)
        x = torch.cat([x, passthrough], dim=1)

        return self.conv8(x)

    @staticmethod
    def __load_yolo_head(yolo_head: DarknetYoloV2):
        for param in yolo_head.parameters():
            param.requires_grad = False

        return yolo_head


if __name__ == '__main__':
    """
    ================================================================
    Total params: 48,529,719
    Trainable params: 28,705,143
    Non-trainable params: 19,824,576
    ----------------------------------------------------------------
    Input size (MB): 1.98
    Forward/backward pass size (MB): 512.27
    Params size (MB): 185.13
    Estimated Total Size (MB): 699.37
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (3, 416, 416)
    darknet = DarknetYoloV2()
    summary(YoloV2(darknet), input_size, device='cpu')
