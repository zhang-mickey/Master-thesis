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
from lib.network.AffinityNet.resnet38_cls import ResNet38_cls
from lib.network.swin_transformer_v2 import *
from lib.network.SERT import *
from lib.network.AffinityNet.resnet38d import *
from lib.network.vgg16d import convert_caffe_to_torch
from lib.network.AffinityNet import *
from lib.network.mix_transformer import *
from lib.network.FickleNet import *
from lib.network.ToCo_model import *


def choose_backbone(backbone_name, pretrained=True, num_classes=1):
    # end to end WSSS
    if backbone_name == 'ToCo':
        net = ToCo_network(
            "deit_base_patch16_224",
            num_classes=num_classes,
            pretrained=pretrained,
            init_momentum=0.9,
            aux_layer=9
        )
        return net
    # fully supervised segmentation
    elif backbone_name == 'deeplabv3plus_resnet50':
        net = DeepLabV3Plus(pretrained=True)
        return net
    elif backbone_name == 'deeplabv3plus_resnet101':
        net = DeepLabV3Plus(pretrained=True)
        return net
    elif backbone_name == 'Segmenter':
        net = Segmenter(num_classes=num_classes)
        return net
    elif backbone_name == 'SERT':
        # (512 / 32) x (512 / 32) = 16 x 16 = 256 patches
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

    # Refinement
    elif backbone_name == 'AffinityNet':
        backbone = resnet101(pretrained=True, num_classes=num_classes)
        affinity_net = AffinityNet(backbone)
        return affinity_net

    # Classification
    elif backbone_name == 'resnet38d':
        net = ResNet38_cls()
        weights_dict = convert_mxnet_to_torch("./model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params")
        net.load_state_dict(weights_dict, strict=False)
        return net
    elif backbone_name == 'vgg16d':
        weights_dict = convert_caffe_to_torch("./model/vgg16_20M.caffemodel")
        net = vgg16d()
        net.load_state_dict(weights_dict, strict=False)
        return net

    elif backbone_name == 'transformer':
        net = vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return net

    elif backbone_name == 'mix_transformer':

        model = mit_b5()

        model_dict = model.state_dict()
        checkpoint = torch.load('./pretrained/mit_b5.pth', map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        else:
            checkpoint = checkpoint
        # for k in ['head.weight', 'head.bias']:
        #     print(f"Removing key {k} from pretrained checkpoint")
        #     del checkpoint[k]
        # for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
        #     print(f"Removing key {k} from pretrained checkpoint")
        #     del checkpoint[k]

        checkpoint.pop('head.weight')
        checkpoint.pop('head.bias')

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

    elif backbone_name == 'conformer':
        model = Net_sm()
        checkpoint = torch.load("./pretrained/transcam_6485.pth", map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        else:
            checkpoint = checkpoint
        model_dict = model.state_dict()
        for k in ['trans_cls_head.weight', 'trans_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    elif backbone_name == 'deeplabv3plus_Xception':  # Add new option
        net = DeepLabv3_plus(pretrained=True)
        return net

    elif backbone_name == "Swin":
        net = SwinTransformerV2(num_classes=1)
        return net

    elif backbone_name == 'resnet101':
        return resnet101(pretrained=pretrained, num_classes=num_classes)

    elif backbone_name == 'FickleNet':
        return fickleresnet101(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError('Backbone name not recognized')

