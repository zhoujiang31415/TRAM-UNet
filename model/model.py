import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_parts import *
from .unet_utils import up_block, down_block
from .conv_trans_utils import *




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.id = UNet
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)            # 256*256
        x2 = self.down1(x1)         # 128*128
        x3 = self.down2(x2)         # 64*64
        x4 = self.down3(x3)         # 32*32
        x5 = self.down4(x4)         # 16*16
        x = self.up1(x5, x4)        # 32*32
        x = self.up2(x, x3)         # 64*64
        x = self.up3(x, x2)         # 128*128
        x = self.up4(x, x1)         # 256*256
        logits = self.outc(x)       # 256*256
        return logits


        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y

