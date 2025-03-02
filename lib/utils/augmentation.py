#both the image and its corresponding mask must be transformed identically

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random


class SegmentationTransform:
    def __init__(self, image_size=512):
        self.image_size = image_size

        # Define transformations for images only
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, mask):
        # Convert images & masks to NumPy (if needed)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Random Horizontal Flip (Apply to both)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random Rotation (-15° to +15°) (Apply to both)
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)  # Nearest neighbor for masks

        # Resize (Apply to both)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Apply image-only transforms
        image = self.image_transforms(transforms.ToPILImage()(image))

        # Convert mask to PyTorch tensor (no normalization)
        mask = torch.from_numpy(mask).long()  # Ensure mask is long type for cross-entropy loss

        return image, mask