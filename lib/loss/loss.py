import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()  # Define the loss function

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)