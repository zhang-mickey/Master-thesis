import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.network.AffinityNet.resnet38d import *

# for classification purposes

class ResNet38_cls(Net):
    def __init__(self):
        super().__init__()
        # Adds a dropout layer
        self.dropout7 = torch.nn.Dropout2d(0.5)
        #Adds a final convolution layer
        self.fc8 = nn.Conv2d(4096, 1, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]


    def forward(self, x):
        #Passes input x through the parent (ResNet38) forward method.
        x = super().forward(x)
        x = self.dropout7(x)

        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        print("resnet38_cls shape",x.shape)
        return x

    def forward_cam(self, x):
        x = super().forward(x)
        #pplies the final convolutional layer ( fc8 ) as a regular convolution (not after pooling),
        #followed by a ReLU activation.
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups