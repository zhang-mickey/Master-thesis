import torch
import numpy as np


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
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
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


#calculating the total number of correct pixels across all images in the batch
def calculate_accuracy(outputs, labels):
    preds = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to binary predictions
    # print(f"Outputs shape: {outputs.shape}")
    # print(f"Labels shape: {labels.shape}")
    correct = (preds == labels).sum().item()
    # print("correct predictions",correct)
    # total = labels.size(0) #the size of the first dimension (batch size)
    total_pixels = labels.numel() 
    # print("labels",total_pixels)
    return (correct / total_pixels)*100


def calculate_iou(outputs, masks, smooth=1e-6):
    pred = (torch.sigmoid(outputs) > 0.5).float()
    intersection = (pred * masks).sum()
    union = pred.sum() + masks.sum() - intersection
    # print(f"Pred positive pixels: {pred.sum().item()}")
    # print(f"Mask positive pixels: {masks.sum().item()}")
    # print(f"Intersection: {intersection.item()}")
    # print(f"Union: {union.item()}")
    return (intersection + smooth) / (union + smooth)


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