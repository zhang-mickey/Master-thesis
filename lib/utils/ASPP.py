import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply convolutions
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        conv7_out = self.conv7(x)

        # Pooling layer (global context)
        pooled_out = self.pool(x)
        pooled_out = self.pool_conv(pooled_out)
        pooled_out = nn.functional.interpolate(pooled_out, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate all outputs
        out = torch.cat([conv1_out, conv3_out, conv5_out, conv7_out, pooled_out], dim=1)

        # Final convolution
        out = self.final_conv(out)

        return out