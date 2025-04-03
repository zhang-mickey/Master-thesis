# both the image and its corresponding mask must be transformed identically

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from PIL import Image


class ColorJitterTransform:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        """
        Applies color jitter only to the image (not mask) with probability p
        """
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = self.jitter(image)
        return image


class CenterCrop_For_Segmentation:
    def __init__(self, crop_size):
        self.crop_size = crop_size  # Can be int (square) or tuple (h, w)

    def __call__(self, image, mask):
        """Crop center from image-mask pair"""
        img_width, img_height = image.size

        # Handle different crop size formats
        if isinstance(self.crop_size, int):
            crop_h = crop_w = self.crop_size
        else:
            crop_h, crop_w = self.crop_size

        # Safety check for small images
        crop_w = min(img_width, crop_w)
        crop_h = min(img_height, crop_h)

        # Calculate center coordinates
        x_center = (img_width - crop_w) // 2
        y_center = (img_height - crop_h) // 2

        # Apply identical crop to both image and mask
        image = image.crop((
            x_center,
            y_center,
            x_center + crop_w,
            y_center + crop_h
        ))

        mask = mask.crop((
            x_center,
            y_center,
            x_center + crop_w,
            y_center + crop_h
        ))

        return image, mask


class RandomCrop_For_Segmentation:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        w, h = image.size
        crop_h, crop_w = self.crop_size, self.crop_size

        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)
        # generate random crop coordinates
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        image = image.crop((x, y, x + crop_w, y + crop_h))
        mask = mask.crop((x, y, x + crop_w, y + crop_h))

        return image, mask


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