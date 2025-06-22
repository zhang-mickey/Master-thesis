import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, stride=None, num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3])

        # PCM
        self.conv6 = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 512 * block.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True)
        )
        self.f8_3 = nn.Conv2d(256 * block.expansion, 256, 1)
        self.f8_4 = nn.Conv2d(512 * block.expansion, 256, 1)
        self.f9 = nn.Conv2d(256 + 256 + 3, 512, 1)
        self.fc8 = nn.Conv2d(512 * block.expansion, num_classes + 1, 1)

        for m in [self.f8_3, self.f8_4, self.f9, self.conv6, self.fc8]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f))  # [n, h*w, h*w]
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    def get_heatmaps(self):
        return self.featmap.clone().detach()

    def forward(self, x):
        d = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)  # conv2
        f2 = self.layer2(f1)  # conv3
        f3 = self.layer3(f2)  # conv4
        f4 = self.layer4(f3)  # conv5
        conv6 = self.conv6(f4)  # 新增的conv6层

        # PCM处理
        n, c, h, w = conv6.size()
        cam = self.fc8(conv6)

        #
        # x_s: torch.Size([8, 64, 32, 32])
        # f8_3: torch.Size([8, 256, 32, 32])
        # f8_4: torch.Size([8, 256, 32, 32])
        f8_3 = self.f8_3(f3)
        f8_4 = self.f8_4(f4)
        x_s = F.interpolate(d, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        f = self.f9(f)
        cam_att = self.PCM(conv6, f)
        cam_att = self.fc8(cam_att)

        featmap = cam + cam_att
        # 分类预测
        pred = F.adaptive_avg_pool2d(featmap[:, :-1], (1, 1)).view(n, -1)

        return pred, featmap, f2  # 返回PCM特征和中间特征


# class ResNet(nn.Module):
#     def __init__(self, block, layers, stride=None):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3])

#         #classification head
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, 1000)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         f1 = self.layer1(x)
#         f2 = self.layer2(f1)
#         f3 = self.layer3(f2)
#         f4 = self.layer4(f3)

#         x = self.avgpool(f4)
#         x = x.view(x.size(0), -1)

#         embedded = x

#         x = self.fc(x)
#         # Returns both the feature embedding and the classification output
#         return x,embedded,[f2,f3,f4]


#     def get_parameter_groups(self, print_fn=print):
#         groups = ([], [], [], [])

#         for name, value in self.named_parameters():
#             # pretrained weights
#             if 'fc' not in name:
#                 if 'weight' in name:
#                     print_fn(f'pretrained weights : {name}')
#                     groups[0].append(value)
#                 else:
#                     print_fn(f'pretrained bias : {name}')
#                     groups[1].append(value)

#             # scracthed weights
#             else:
#                 if 'weight' in name:
#                     if print_fn is not None:
#                         print_fn(f'scratched weights : {name}')
#                     groups[2].append(value)
#                 else:
#                     if print_fn is not None:
#                         print_fn(f'scratched bias : {name}')
#                     groups[3].append(value)
#         return groups


def resnet18(pretrained=False, stride=None, num_classes=1, **kwargs):
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(BasicBlock, [2, 2, 2, 2], stride=stride, **kwargs)
    if pretrained:
        model.backbone.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=True)
    model.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    return model


def resnet34(pretrained=False, stride=None, num_classes=1, **kwargs):
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(BasicBlock, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        model.backbone.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    model.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    return model


def resnet50(pretrained=False, stride=None, num_classes=1, **kwargs):
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


def resnet101(pretrained=False, stride=None, num_classes=1, **kwargs):
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 23, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model


def resnet152(pretrained=False, stride=None, num_classes=1, **kwargs):
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 8, 36, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=True)
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    return model



