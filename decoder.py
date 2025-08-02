import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Conv → BatchNorm → ReLU
        self.relu = nn.ReLU(inplace=True) # 共用

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.relu(self.norm1(self.conv1(x)))

        x = self.relu(self.norm2(self.deconv2(x)))

        x = self.relu(self.norm3(self.conv3(x)))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder1 = DecoderBlock(in_channels=516, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=512,out_channels=128)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=64)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, features):

        # from encoder [e1, e2, e3, e4]
        # Feature 1 shape: torch.Size([1, 64, 112, 112]) --> e1
        # Feature 2 shape: torch.Size([1, 128, 56, 56]) --> e2
        # Feature 3 shape: torch.Size([1, 256, 28, 28]) --> e3
        # Feature 4 shape: torch.Size([1, 516, 14, 14]) --> e4

        e1, e2, e3, e4 = features

        f4 = self.decoder1(e4) # 516 --> 256
        f3 = self.decoder2((torch.cat([f4, e3], dim=1))) # 516 --> 128
        f2 = self.decoder3(torch.cat([f3, e2], dim=1)) # 256 --> 64
        f1 = self.decoder4(torch.cat([f2, e1], dim=1)) # 128 --> 64

        out = self.finaldeconv1(f1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
 
        return out

