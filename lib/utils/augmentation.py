# both the image and its corresponding mask must be transformed identically

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
from albumentations import Compose
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout


class SmokeCopyPaste:
    def __init__(self, smoke_dataset, p=0.7, max_objects=1):
        """
        Args:
            smoke_dataset: Dataset containing real smoke images/masks
            p: Probability of applying this augmentation
            max_objects: Max smoke objects to paste per image
        """
        self.smoke_data = [(img, mask) for img, mask in smoke_dataset]
        self.p = p
        self.max_objects = max_objects
        self.aug = Compose([
            CoarseDropout(max_holes=5, max_height=64, max_width=64, fill_value=0, p=0.3)
        ])

    def __call__(self, non_smoke_image, label):
        if label == 1 or random.random() > self.p:
            return non_smoke_image, torch.tensor(0.0)  # No augmentation, return as non-smoke

        # Convert to OpenCV format
        non_smoke_image = np.array(non_smoke_image)

        num_objects = random.randint(1, self.max_objects)
        for _ in range(num_objects):
            smoke_img, smoke_mask = random.choice(self.smoke_data)

            # Convert smoke image to numpy array if it's not already
            if isinstance(smoke_img, Image.Image):
                smoke_img = np.array(smoke_img)
            elif isinstance(smoke_img, torch.Tensor):
                smoke_img = smoke_img.cpu().numpy()

            # Convert smoke mask to binary
            if isinstance(smoke_mask, Image.Image):
                smoke_mask = np.array(smoke_mask)
            elif isinstance(smoke_mask, torch.Tensor):
                smoke_mask = smoke_mask.cpu().numpy()

            smoke_mask = (smoke_mask > 0).astype(np.uint8)
            h, w = smoke_mask.shape

            # Check if smoke mask is larger than the non-smoke image and resize if needed
            if h >= non_smoke_image.shape[0] or w >= non_smoke_image.shape[1]:
                # Resize smoke mask and image to fit within the non-smoke image
                scale = min(non_smoke_image.shape[0] / (h + 1), non_smoke_image.shape[1] / (w + 1))
                new_h, new_w = int(h * scale), int(w * scale)

                if new_h <= 0 or new_w <= 0:
                    # Skip this smoke object if it can't be resized properly
                    continue

                smoke_mask = cv2.resize(smoke_mask, (new_w, new_h))
                smoke_img = cv2.resize(smoke_img, (new_w, new_h))
                h, w = new_h, new_w

            # Random position for pasting (now safe because h < image height)
            y_offset = random.randint(0, max(0, non_smoke_image.shape[0] - h))
            x_offset = random.randint(0, max(0, non_smoke_image.shape[1] - w))

            # Extract ROI from non-smoke image
            roi = non_smoke_image[y_offset:y_offset + h, x_offset:x_offset + w].copy()

            # Ensure ROI and smoke image have compatible shapes
            if roi.shape[:2] != smoke_img.shape[:2]:
                # Adjust smoke_img to match ROI shape
                smoke_img = cv2.resize(smoke_img, (roi.shape[1], roi.shape[0]))
                smoke_mask = cv2.resize(smoke_mask, (roi.shape[1], roi.shape[0]))
                h, w = roi.shape[:2]

            # Ensure both images have the same number of channels
            if len(smoke_img.shape) == 2:  # Grayscale
                smoke_img = cv2.cvtColor(smoke_img, cv2.COLOR_GRAY2BGR)
            if len(roi.shape) == 2:  # Grayscale
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

            try:
                # Blend the smoke mask onto non-smoke background
                blended = cv2.seamlessClone(
                    smoke_img, roi, (smoke_mask * 255).astype(np.uint8),
                    (w // 2, h // 2), cv2.NORMAL_CLONE
                )

                # Place blended result back
                non_smoke_image[y_offset:y_offset + h, x_offset:x_offset + w] = blended
            except Exception as e:
                print(f"Seamless cloning failed: {e}")
                # Fallback to simple alpha blending
                alpha = 0.7
                mask_3ch = np.stack([smoke_mask] * 3, axis=2) * alpha
                blended = roi * (1 - mask_3ch) + smoke_img * mask_3ch
                non_smoke_image[y_offset:y_offset + h, x_offset:x_offset + w] = blended

        return Image.fromarray(non_smoke_image), torch.tensor(1.0)  # Now labeled as smoke


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
    def __init__(self, crop_size, p=0.5):
        self.crop_size = crop_size  # Can be int (square) or tuple (h, w)
        self.p = p

    def __call__(self, image, mask):
        """Crop center from image-mask pair"""
        if random.random() < self.p:
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
        else:
            return image, mask


class RandomCrop_For_Segmentation:
    def __init__(self, crop_size, p=0.5):
        self.p = p
        self.crop_size = crop_size

    def __call__(self, image, mask):
        if random.random() < self.p:
            return image, mask
        else:

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