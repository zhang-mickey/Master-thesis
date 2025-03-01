import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()  # Define the loss function

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)