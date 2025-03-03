import torch
import numpy as np

def compute_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()  # Convert logits to binary mask
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

    dice = (2 * intersection) / (pred.sum() + target.sum() + 1e-6)

    accuracy = (pred == target).float().mean()

    return iou.item(), dice.item(), accuracy.item()



def calculate_accuracy(outputs, masks):
    pred = outputs > 0.5  # Apply threshold
    correct = (pred == masks).float().sum()
    return correct / masks.numel()  # Total number of elements

def calculate_iou(outputs, masks, smooth=1e-6):
    pred = (outputs > 0.5).float()
    intersection = (pred * masks).sum()
    union = pred.sum() + masks.sum() - intersection
    return (intersection + smooth) / (union + smooth)
