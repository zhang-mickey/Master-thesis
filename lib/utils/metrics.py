import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


def compute_similarity(vit, resnet, metric='cosine', mode='channel', batch_idx=0):
    if metric == 'cosine' and mode == 'channel':
        vit = vit.view(vit.size(0), vit.size(1), -1)
        resnet = resnet.view(resnet.size(0), resnet.size(1), -1)
        sim = F.cosine_similarity(vit.unsqueeze(2), resnet.unsqueeze(1), dim=3)

        cos_sim_np = sim.detach().cpu().numpy().flatten()
        plt.figure(figsize=(6, 4))
        plt.hist(cos_sim_np, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Channel-wise Cosine Similarity\n({batch_idx})")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.savefig(f"cosine_sim_{batch_idx}.png")
        plt.close()
        return 1 - sim.mean()

    elif metric == 'cosine' and mode == 'global':
        if vit.shape[1] != resnet.shape[1]:
            # Linear expects [B, L, C], so permute first
            resnet = resnet.permute(0, 2, 1)  # [B, L, C_in]
            proj = nn.Linear(resnet.shape[-1], vit.shape[1]).to(resnet.device)  # C_in -> C_out
            resnet = proj(resnet)  # [B, L, C_out]
            resnet = resnet.permute(0, 2, 1)  # [B, C_out, L]
        vit = vit.reshape(vit.size(0), -1)
        resnet = resnet.reshape(resnet.size(0), -1)
        sim = F.cosine_similarity(vit, resnet, dim=1)
        return 1 - sim.mean()
    elif metric == 'cosine' and mode == 'spatial':

        vit = vit.permute(0, 2, 1)  # [B, L, C1]
        resnet = resnet.permute(0, 2, 1)  # [B, L, C2]
        if vit.shape[2] != resnet.shape[2]:
            proj = nn.Linear(resnet.shape[-1], vit.shape[-1]).to(resnet.device)
            resnet = proj(resnet)

        sim = F.cosine_similarity(vit, resnet, dim=-1)  # [B, L] every patch
        return 1 - sim.mean()

    elif mode == 'spatial_map':
        vit_map = vit.mean(dim=1)  # [B, H, W]
        resnet_map = resnet.mean(dim=1)  # [B, H, W]

        vit_map_flat = vit_map.view(vit_map.size(0), -1)
        resnet_map_flat = resnet_map.view(resnet_map.size(0), -1)

        if metric == 'cosine':
            sim = F.cosine_similarity(vit_map_flat, resnet_map_flat, dim=1)
            return 1 - sim.mean()
        elif metric == 'l1':
            return torch.abs(vit_map_flat - resnet_map_flat).mean()
        elif metric == 'l2':
            return torch.norm(vit_map_flat - resnet_map_flat, p=2, dim=1).mean()

    elif metric == 'inner' and mode == 'channel':
        # flatten spatial dims
        vit = vit.view(vit.size(0), vit.size(1), -1)  # [B, C, H*W]
        resnet = resnet.view(resnet.size(0), resnet.size(1), -1)

        # compute Gram matrix [B, C, C]
        vit_gram = torch.bmm(vit, vit.transpose(1, 2))  # [B, C, C]
        resnet_gram = torch.bmm(resnet, resnet.transpose(1, 2))  # [B, C, C]

        # normalize
        # vit_gram = F.normalize(vit_gram.view(vit.size(0), -1), dim=1)
        # resnet_gram = F.normalize(resnet_gram.view(resnet.size(0), -1), dim=1)

        sim = F.cosine_similarity(vit_gram, resnet_gram, dim=1)  # [B]
        return 1 - sim.mean()

    elif metric == 'l1' and mode == 'global':
        vit_flat = vit.reshape(vit.size(0), -1)
        resnet_flat = resnet.reshape(resnet.size(0), -1)
        if vit_flat.shape[1] != resnet_flat.shape[1]:
            proj = nn.Linear(resnet_flat.shape[1], vit_flat.shape[1]).to(resnet.device)
            resnet_flat = proj(resnet_flat)
        return torch.abs(vit_flat - resnet_flat).mean()

    elif metric == 'l2' and mode == 'global':
        vit_flat = vit.reshape(vit.size(0), -1)
        resnet_flat = resnet.reshape(resnet.size(0), -1)
        if vit_flat.shape[1] != resnet_flat.shape[1]:
            proj = nn.Linear(resnet_flat.shape[1], vit_flat.shape[1]).to(resnet.device)
            resnet_flat = proj(resnet_flat)
        return torch.norm(vit_flat - resnet_flat, p=2, dim=1).mean()

    elif mode == 'logits':
        if metric == 'cosine':
            sim = F.cosine_similarity(vit, resnet, dim=1)
            return 1 - sim.mean()
        elif metric == 'kl':
            vit_soft = F.log_softmax(vit, dim=1)
            resnet_soft = F.softmax(resnet, dim=1)
            return F.kl_div(vit_soft, resnet_soft, reduction='batchmean')
        elif metric == 'l1':
            return torch.abs(vit - resnet).mean()
        elif metric == 'l2':
            return torch.norm(vit - resnet, p=2, dim=1).mean()
        else:
            raise ValueError(f"Unsupported metric '{metric}' for logits")


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