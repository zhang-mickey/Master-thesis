import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(img_size=512):
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),

        transforms.ToTensor(),  # Convert to tensor
        # transforms.Lambda(lambda x: torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0)))  # Convert to binary mask
        transforms.Lambda(lambda x: (x > 0).float().squeeze(0))
    ])

    return image_transform, mask_transform