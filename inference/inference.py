import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

def inference_dataset():
    def __init__(self,):

        self.transform=transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to match model input
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
])


