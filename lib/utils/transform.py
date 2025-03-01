import numpy as np
import torch
from torchvision import transforms
# ---- Define Transforms ----

def get_transforms(img_size=512):
    """Returns transformation functions for images and masks.

    Args:
        img_size (int): The size to which images and masks should be resized.

    Returns:
        tuple: (image_transform, mask_transform)
    """
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to match model input
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to match image size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0)))  # Convert to binary mask
    ])

    return image_transform, mask_transform