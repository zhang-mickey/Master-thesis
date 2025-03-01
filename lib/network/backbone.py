import torch
import torch.nn as nn
from torchvision import models
from lib.network.DeepLabV3Plus import DeepLabV3Plus
from lib.network.xception import xception


def choose_backbone(backbone_name,pretrained=True,num_classes=1):

    if backbone_name == 'deeplabv3plus_resnet50':
        net =DeepLabV3Plus(pretrained=True)
        return  net

    elif backbone_name == 'xception':
        net=xception(pretrained=pretrained)
        return net
    else:
        raise ValueError('Backbone name not recognized')
