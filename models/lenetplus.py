from torch import nn
import torch.nn.functional as F

"""
LeNetPlus changed from LeNet
"""


def _make_conv_block(in_channels, out_channels, num_layer=2, stride=1, pad=2):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=pad))
    layers.append(nn.LeakyReLU(0.2))

    for _ in range(num_layer - 1):
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=pad))
        layers.append(nn.LeakyReLU(0.2))

    layers.append(nn.MaxPool2d(2))
    out = nn.Sequential(*layers)

    return out


class L2Normalization(nn.Module):
    """Applies L2 Normalization to input.
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, mode, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def forward(self, in_features):
        return nn.functional.normalize(in_features)


class LeNetPlus(nn.Module):
    """
    LeNetPlus model
    """
    def __init__(self, classes=10, feature_size=256, use_dropout=True, use_norm=False, use_bn=False, use_inn=False,
                 use_angular=False, **kwargs):
        super(LeNetPlus, self).__init__(**kwargs)
        in_out = [(3, 32), (32, 64), (64, 128)]

        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.use_bn = use_bn
        self.use_inn = use_inn
        self.use_angular = use_angular
        self.layers = []

        if self.use_inn:
            self.layers.append(nn.InstanceNorm2d(3))

        for i, chan in enumerate(in_out):
            in_channels, out_channels = chan
            if use_bn:
                self.layers.append(nn.BatchNorm2d(in_channels))

            self.layers.append(_make_conv_block(in_channels, out_channels))

            if use_dropout and i > 0:
                self.layers.append(nn.Dropout(0.5))

        if self.use_norm:
            self.layers.append(L2Normalization())

        self.layers.append(nn.modules.Flatten())
        self.net = nn.Sequential(*self.layers)
        self.output = nn.Linear(6272, classes)

    def forward(self, in_features):
        out = self.net(in_features)
        out = self.output(out)
        out = F.softmax(out, dim=1)
        return out
