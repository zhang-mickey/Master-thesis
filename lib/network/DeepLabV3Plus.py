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
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extract layers for different feature levels
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # Low-level features
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # High-level features
        
        # ASPP module for high-level features
        self.aspp = ASPPModule(in_channels=2048, out_channels=256)
        
        # Low-level features processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Decoder module
        self.decoder = DecoderModule(in_channels_high=256, in_channels_low=48, num_classes=num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Forward pass through backbone layers
        x0 = self.layer0(x)
        x1 = self.layer1(x0)  # Low-level features
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # High-level features
        
        # Print shapes for debugging
        # print(f"Input shape: {x.shape}")
        # print(f"Low-level features shape (x1): {x1.shape}")
        # print(f"High-level features shape (x4): {x4.shape}")
        
        # Apply ASPP to high-level features
        aspp_out = self.aspp(x4)
        # print(f"ASPP output shape: {aspp_out.shape}")
        
        # Process low-level features
        low_level_features = self.low_level_conv(x1)
        # print(f"Processed low-level features shape: {low_level_features.shape}")
        
        # Decode and upsample
        output = self.decoder(aspp_out, low_level_features)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output


class DecoderModule(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, num_classes):
        super(DecoderModule, self).__init__()
        
        # Instead of fixed scale factor, we'll use dynamic upsampling
        # to match the low-level feature dimensions
        
        # Fusion of high and low level features
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_high + in_channels_low, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, high_level_features, low_level_features):
        # Get the spatial dimensions of the low-level features
        low_level_size = low_level_features.size()[2:]
        
        # Upsample high-level features to match low-level features size
        high_level_features = F.interpolate(
            high_level_features, 
            size=low_level_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Print shapes for debugging
        # print(f"Upsampled high-level features shape: {high_level_features.shape}")
        # print(f"Low-level features shape: {low_level_features.shape}")
        
        # Concatenate with low-level features
        x = torch.cat([high_level_features, low_level_features], dim=1)
        
        # Apply fusion convolutions
        x = self.fusion(x)
        
        return x

