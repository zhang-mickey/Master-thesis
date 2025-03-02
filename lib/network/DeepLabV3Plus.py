import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

from lib.utils.ASPP import ASPPModule

#DeepLabV3+ has a decoder and Atrous Spatial Pyramid Pooling (ASPP)
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        # 1. Backbone: ResNet50 for feature extraction
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers (fully connected)

        # 2. Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)

        # 3. Decoder
        self.decoder = DecoderModule(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        # Forward pass through backbone
        features = self.backbone(x)
        # print(f"Features shape before ASPP: {features.shape}")  # Check shape
        # Apply ASPP to features
        aspp_out = self.aspp(features)
        # print(f"Features shape after ASPP: {aspp_out.shape}")  # Check shape

        # Manually upsample the output to match the input image size (512x512)
        upsampled_out = F.interpolate(aspp_out, size=x.size()[2:], mode='bilinear', align_corners=False)
        # print(f"Features shape after upsample: {aspp_out.shape}")  # Check shape
        output = self.decoder(upsampled_out)
        # Decode output to segmentation map
        # output = self.decoder(aspp_out)

        return output





class DecoderModule(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DecoderModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x):
        # Decoder layers
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)

        return x

