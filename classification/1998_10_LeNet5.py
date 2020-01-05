from collections import OrderedDict

import torch.nn as nn

from classification.utils import Flatten


class LeNet5(nn.Module):
    """ Gradient-Based Learning Applied to Document Recognition
        http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = self.build_feature_extractor()
        self.classifier = self.build_classifier()

    @staticmethod
    def build_feature_extractor():
        C1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1, padding=2, bias=True)
        S2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv1d(6, 6, kernel_size=(1, 1), bias=True),
            nn.Sigmoid()
        )
        C3 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        S4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv1d(16, 16, kernel_size=(1, 1), bias=True),
            nn.Sigmoid()
        )
        C5 = nn.Conv2d(16, 120, kernel_size=(5, 5), stride=1, padding=0, bias=True)

        return nn.Sequential(OrderedDict([
            ('C1', C1),
            ('S2', S2),
            ('C3', C3),
            ('S4', S4),
            ('C5', C5)
        ]))

    @staticmethod
    def build_classifier():
        F6 = nn.Sequential(
            Flatten(),
            nn.Linear(120, 84),
            nn.Tanh()
        )

        F7 = nn.Sequential(
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )

        return nn.Sequential(OrderedDict([
            ('F6', F6),
            ('F7', F7)
        ]))

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)


if __name__ == '__main__':
    """
    ================================================================
    Total params: 62,020
    Trainable params: 62,020
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.09
    Params size (MB): 0.24
    Estimated Total Size (MB): 0.33
    ----------------------------------------------------------------
    """
    from torchsummary import summary

    input_size = (1, 28, 28)
    summary(LeNet5(), input_size, device='cpu')
