import torch
import numpy as np
import torch.nn.functional as F


def linear_CKA(X, Y):
    """
    Compute linear CKA between X and Y.
    X: [B, C, D] — e.g., ViT feature maps
    Y: [B, C, D] — e.g., ResNet feature maps
    """
    # Flatten over spatial dim: [B, C, D] → [B, C×D]
    X = X.flatten(1)
    Y = Y.flatten(1)

    # Center each feature
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute dot products
    dot_XY = (X * Y).sum()
    norm_X = torch.norm(X, p='fro') ** 2
    norm_Y = torch.norm(Y, p='fro') ** 2

    cka = dot_XY ** 2 / (norm_X * norm_Y + 1e-8)
    return cka


def compute_similarity(vit, resnet, metric='cosine'):
    vit = vit.view(vit.size(0), vit.size(1), -1)
    resnet = resnet.view(resnet.size(0), resnet.size(1), -1)

    if metric == 'cosine':
        sim = F.cosine_similarity(vit.unsqueeze(2), resnet.unsqueeze(1), dim=3)
        return 1 - sim.mean()

    elif metric == 'cka':
        return 1 - linear_CKA(vit, resnet)  # 返回 dissimilarity

    elif metric == 'euclidean':
        diff = vit.unsqueeze(2) - resnet.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=3)
        return dist.mean()

    elif metric == 'mse':
        diff = vit.unsqueeze(2) - resnet.unsqueeze(1)
        mse = (diff ** 2).mean(dim=-1)
        return mse.mean()

    elif metric == 'pearson':
        vit_centered = vit.unsqueeze(2) - vit.unsqueeze(2).mean(dim=3, keepdim=True)
        resnet_centered = resnet.unsqueeze(1) - resnet.unsqueeze(1).mean(dim=3, keepdim=True)
        numerator = (vit_centered * resnet_centered).sum(dim=3)
        denominator = torch.sqrt((vit_centered ** 2).sum(dim=3) * (resnet_centered ** 2).sum(dim=3) + 1e-8)
        sim = numerator / (denominator + 1e-8)
        return 1 - sim.mean()

    else:
        raise ValueError(f"Unknown metric: {metric}")


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] if self.__data[k][1] > 0 else 0.0 for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


# def compute_metrics(pred, target, threshold=0.5):
#     pred = (pred > threshold).float()  # Convert logits to binary mask
#     target = target.float()

#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum() - intersection
#     iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

#     dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-6)

#     accuracy = (pred == target).float().mean()

#     return iou.item(), dice.item(), accuracy.item()


# calculating the total number of correct pixels across all images in the batch
def calculate_accuracy(outputs, labels):
    if outputs.dtype.is_floating_point and outputs.max() > 1:
        preds = (torch.sigmoid(outputs) > 0.5).float()
    else:
        preds = (outputs > 0.5).float()  # Convert logits to binary predictions
    # print(f"Outputs shape: {outputs.shape}")
    # print(f"Labels shape: {labels.shape}")
    correct = (preds == labels).sum().item()
    # print("correct predictions",correct)
    # total = labels.size(0) #the size of the first dimension (batch size)
    total_pixels = labels.numel()
    # print("labels",total_pixels)
    return (correct / total_pixels) * 100


def calculate_f1_score(outputs, labels, beta=1.0, smooth=1e-8):
    # Flatten all elements to 1D
    outputs = outputs.view(-1)
    labels = labels.view(-1)

    # Convert logits to binary predictions
    if outputs.dtype.is_floating_point and outputs.max() > 1:
        preds = (torch.sigmoid(outputs) > 0.5).float()
    else:
        preds = (outputs > 0.5).float()

    labels = labels.float()

    # Compute TP, FP, FN
    TP = (preds * labels).sum()
    FP = (preds * (1 - labels)).sum()
    FN = ((1 - preds) * labels).sum()

    # Compute precision and recall
    precision = TP / (TP + FP + smooth) * 100
    recall = TP / (TP + FN + smooth) * 100

    # Compute F-beta score
    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + smooth)

    return f1.item()


def calculate_iou(outputs, masks, smooth=1e-6):
    if outputs.dtype.is_floating_point and outputs.max() > 1:
        pred = (torch.sigmoid(outputs) > 0.5).float()
    else:
        pred = (outputs > 0.5).float()
    intersection = (pred * masks).sum()
    union = pred.sum() + masks.sum() - intersection
    # print(f"Pred positive pixels: {pred.sum().item()}")
    # print(f"Mask positive pixels: {masks.sum().item()}")
    # print(f"Intersection: {intersection.item()}")
    # print(f"Union: {union.item()}")

    # return (intersection + smooth) / (union + smooth)
    iou = intersection / union if union != 0 else 0
    return iou


# def calculate_iou(outputs, masks, smooth=1e-6):
#     pred = (torch.sigmoid(outputs) > 0.5).float()
#     intersection = (pred * masks).sum(dim=(1, 2))  # 计算 batch 维度上的 IoU
#     union = pred.sum(dim=(1, 2)) + masks.sum(dim=(1, 2)) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     return iou.sum().item()


def calculate_f1(output, target):
    precision = calculate_precision(output, target)
    recall = calculate_recall(output, target)
    return 2 * (precision * recall) / (precision + recall + 1e-6)


def calculate_precision(output, target):
    # Convert to binary using threshold, then to boolean for bitwise operations
    output = (torch.sigmoid(output) > 0.5).bool()
    target = target.bool()  # Convert target to boolean

    # Calculate true positives and false positives
    true_positives = (output & target).float().sum()
    false_positives = (output & ~target).float().sum()

    return true_positives / (true_positives + false_positives + 1e-6)


def calculate_recall(output, target):
    # Convert to binary using threshold, then to boolean for bitwise operations
    output = (torch.sigmoid(output) > 0.5).bool()
    target = target.bool()  # Convert target to boolean

    # Calculate true positives and false negatives
    true_positives = (output & target).float().sum()
    false_negatives = (~output & target).float().sum()

    return true_positives / (true_positives + false_negatives + 1e-6)


def calculate_dice(output, target, smooth=1e-6):
    # Convert to binary using threshold, then to boolean for bitwise operations
    output = (torch.sigmoid(output) > 0.5).bool()
    target = target.bool()  # Convert target to boolean

    # Calculate intersection
    intersection = (output & target).float().sum()

    return (2. * intersection + smooth) / (output.float().sum() + target.float().sum() + smooth)