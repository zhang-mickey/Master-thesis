import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from lib.network.DeepLabV3Plus import DeepLabV3Plus
from lib.network.resnet import *
from lib.network.VIT import *
from lib.network.ViTSegmenter import *
# from lib.network.ViTSegmenter import create_segmenter
from lib.network.deeplab_xception import *
from lib.network.Conformer import *
from lib.network.swin_transformer_v2 import *
from lib.network.SERT import *


def choose_backbone(backbone_name, pretrained=True, num_classes=1):
    # segmentation
    if backbone_name == 'deeplabv3plus_resnet50':
        net = DeepLabV3Plus(pretrained=True)
        return net
    elif backbone_name == 'deeplabv3plus_resnet101':
        net = DeepLabV3Plus(pretrained=True)
        return net
    elif backbone_name == 'Segmenter':
        net = Segmenter(num_classes=num_classes)
        return net
    elif backbone_name == 'SERT':

        model = SETRModel(patch_size=(32, 32),
                          in_channels=3,
                          out_channels=1,
                          hidden_size=1024,
                          num_hidden_layers=6,
                          num_attention_heads=16,
                          decode_features=[512, 256, 128, 64])
        return model
    # elif backbone_name=='SegFormer':
    #     net =SegFormer()
    #     return  net

    # Classification
    elif backbone_name == 'transformer':
        net = vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return net
    elif backbone_name == 'conformer':
        net = Net_sm()
        return net
    elif backbone_name == 'deeplabv3plus_Xception':  # Add new option
        net = DeepLabv3_plus(pretrained=True)
        return net
    elif backbone_name == "Swin":
        net = SwinTransformerV2(num_classes=1)
        return net

    elif backbone_name == 'resnet18':
        return resnet34(pretrained=pretrained, num_classes=num_classes)
    elif backbone_name == 'resnet34':
        return resnet34(pretrained=pretrained, num_classes=num_classes)
    elif backbone_name == 'resnet50':
        return resnet50(pretrained=pretrained, num_classes=num_classes)
    elif backbone_name == 'resnet101':
        return resnet101(pretrained=pretrained, num_classes=num_classes)
    elif backbone_name == 'resnet152':
        return resnet152(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError('Backbone name not recognized')
