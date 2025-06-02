import torch.nn as nn
from torch.nn import functional as F


# Modified from D2lai: https://d2l.ai/chapter_convolutional-modern/resnet.html
class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(self, in_ch, out_ch, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f"b{i+2}", self.block(*b, first_block=(i == 0)))
        self.net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(arch[-1][-1], num_classes)

    def b1(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def block(self, num_residuals, in_ch, out_ch, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_ch, out_ch, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_ch, out_ch))
        return nn.Sequential(*blk)

    def forward(self, x):

        return self.fc(self.net(x).flatten(start_dim=1))
