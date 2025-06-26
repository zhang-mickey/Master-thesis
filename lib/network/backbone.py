import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from segment_anything import sam_model_registry
from lib.network.DeepLabV3Plus import DeepLabV3Plus
from lib.network.resnet import *
from lib.network.VIT import *
from lib.network.ViTSegmenter import *
# from lib.network.ViTSegmenter import create_segmenter
from lib.network.deeplab_xception import *
from lib.network.Conformer import *
from lib.network.resnet_normal import *
from lib.network.swin_transformer_v2 import *
from lib.network.SERT import *
from lib.network.vgg16d import convert_caffe_to_torch
from lib.network.AffinityNet import *
from lib.network.mix_transformer import *
from lib.network.FickleNet import *
from lib.network.ToCo_model import *
from lib.network.mobilenetv2 import *


class WeakClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        if hasattr(feature_extractor, 'output_channels'):
            in_channels = feature_extractor.output_channels
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 1024, 1024)
                dummy_output = feature_extractor(dummy_input)
                in_channels = dummy_output.shape[1]

        self.classifier = nn.Sequential(
            # 保留空间信息
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 多尺度特征融合
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # 深层分类器
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


def choose_backbone(backbone_name, pretrained=True, num_classes=1):
    # end to end WSSS
    if backbone_name == 'sam':
        sam = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
        image_encoder = sam.image_encoder
        for param in image_encoder.parameters():  # 默认冻结
            param.requires_grad = False
        for name, param in image_encoder.named_parameters():
            if "blocks.23" in name or "blocks.22" in name or "blocks.21" in name:
                param.requires_grad = True

        model = WeakClassifier(image_encoder, num_classes=1)
        return model

    elif backbone_name == 'ToCo':
        net = ToCo_network(
            # "deit_base_patch16_224",
            'vit_base_patch16_224',
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
        backbone = resnet101_raw(pretrained=True, num_classes=num_classes)
        affinity_net = Affinity_Net(backbone)
        return affinity_net

    # Classification
    elif backbone_name == 'vit_s':
        model = backbone_vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        return model

    elif backbone_name == 'vit_b':
        model = backbone_vit_base_patch16_224(pretrained=True, num_classes=num_classes)
        # for param in model.patch_embed.parameters():
        #     param.requires_grad = False
        # for param in model.blocks[0].parameters():
        #     param.requires_grad = False
        return model

    elif backbone_name == 'resnet101_raw':
        return resnet101_raw(pretrained=pretrained, num_classes=num_classes)

    elif backbone_name == 'resnet50_raw':
        return resnet50_raw(pretrained=pretrained, num_classes=num_classes)

    # with PCM
    elif backbone_name == 'resnet101':
        return resnet101(pretrained=pretrained, num_classes=num_classes)

    elif backbone_name == 'resnet50':
        return resnet50(pretrained=pretrained, num_classes=num_classes)

    elif backbone_name == 'mobilenetv2':

        model = MobileNetV2()
        if pretrained:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
            model.load_state_dict(state_dict)
        model.classifier = nn.Linear(model.last_channel, 1)
        return model


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



    elif backbone_name == 'FickleNet':
        return fickleresnet101(pretrained=pretrained, num_classes=num_classes)

    elif backbone_name == 'resnet38d':
        net = res38Net(num_classes=num_classes)
        weights_dict = torch.load("./pretrained/resnet38_dict.pth")
        # weights_dict=convert_mxnet_to_torch("./model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params")
        net.load_state_dict(weights_dict, strict=False)
        return net

    elif backbone_name == 'vgg16d':
        weights_dict = convert_caffe_to_torch("./model/vgg16_20M.caffemodel")
        net = vgg16d()
        net.load_state_dict(weights_dict, strict=False)
        return net
