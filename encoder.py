import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DACBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out1 = self.relu(self.dilate1(x))
        out2 = self.relu(self.conv1x1(self.dilate2(x)))
        out3 = self.relu(self.conv1x1(self.dilate2(self.dilate1(x))))
        out4 = self.relu(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))

        return x + out1 + out2 + out3 + out4


class RMPBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)

    def forward(self, x):

        size = x.size()[2:] # c, h, w   通道 長 寬
        p1 = F.interpolate(self.conv(self.pool1(x)), size=size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.conv(self.pool2(x)), size=size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.conv(self.pool3(x)), size=size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.conv(self.pool4(x)), size=size, mode='bilinear', align_corners=False)

        out = torch.cat([x, p1, p2, p3, p4], 1) # 14*14*(x:512 + 4)

        return out 


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet34(weights=None)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 輸出大小 112x112
        self.encoder1 = resnet.layer1 
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3 
        self.encoder4 = resnet.layer4 
        self.dac = DACBlock(512)  # resnet34 layer4 output is 512 channels
        self.rmp = RMPBlock(512)

    def forward(self, x):

        f1 = self.stage1(x)

        e1 = self.encoder1(f1)  
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dac(e4)
        e4 = self.rmp(e4)
        return [e1, e2, e3, e4]  # return multi-scale features

