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
    def __init__(self, smoke_dataset, p=0.5, max_objects=1):
        self.smoke_data = [(img, mask) for img, mask, _ in smoke_dataset]
        self.p = p
        self.max_objects = max_objects

    def __call__(self, non_smoke_image, label, return_mask=False):
        if label == 1 or random.random() > self.p:
            if return_mask:
                h, w = non_smoke_image.size[1], non_smoke_image.size[0]
                return non_smoke_image, torch.tensor(0.0), torch.zeros((h, w), dtype=torch.uint8)
            return non_smoke_image, torch.tensor(0.0)

        # Convert PIL Image to numpy array (HWC format)
        non_smoke_np = np.array(non_smoke_image)
        if non_smoke_np.ndim == 2:  # Grayscale to RGB
            non_smoke_np = np.stack([non_smoke_np] * 3, axis=-1)
        h, w = non_smoke_np.shape[:2]

        mask_total = np.zeros((h, w), dtype=np.uint8)

        for _ in range(random.randint(1, self.max_objects)):
            smoke_img, smoke_mask = random.choice(self.smoke_data)

            # Convert to numpy arrays
            smoke_img = np.array(smoke_img.convert('RGB')) if isinstance(smoke_img, Image.Image) \
                else smoke_img.cpu().numpy().transpose(1, 2, 0)
            smoke_mask = np.array(smoke_mask) if isinstance(smoke_mask, Image.Image) \
                else smoke_mask.cpu().numpy().squeeze()

            # Resize with aspect ratio preservation
            smoke_h, smoke_w = smoke_mask.shape
            scale = min(h / smoke_h, w / smoke_w)
            new_h, new_w = int(smoke_h * scale), int(smoke_w * scale)

            smoke_img = cv2.resize(smoke_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            smoke_mask = cv2.resize(smoke_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            smoke_mask = (smoke_mask > 127).astype(np.uint8)

            # Random position with boundary checks
            y1 = random.randint(0, h - new_h)
            x1 = random.randint(0, w - new_w)
            y2 = y1 + new_h
            x2 = x1 + new_w

            try:
                blended = cv2.seamlessClone(
                    # the source image
                    smoke_img,
                    # destination region where the smoke will be placed.
                    non_smoke_np[y1:y2, x1:x2],

                    (smoke_mask * 255).astype(np.uint8),

                    (new_w // 2, new_h // 2),
                    cv2.NORMAL_CLONE
                )
            except:
                mask = smoke_mask[..., np.newaxis].astype(np.float32)
                non_smoke_roi = non_smoke_np[y1:y2, x1:x2].astype(np.float32)
                blended = non_smoke_roi * (1 - mask) + smoke_img.astype(np.float32) * mask
                blended = np.clip(blended, 0, 255).astype(np.uint8)

            non_smoke_np[y1:y2, x1:x2] = blended
            mask_total[y1:y2, x1:x2] = np.maximum(mask_total[y1:y2, x1:x2], smoke_mask)

        # Convert back to PIL Image
        result_image = Image.fromarray(non_smoke_np)
        if return_mask:
            return result_image, torch.tensor(1.0), mask_total
        return result_image, torch.tensor(1.0)


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
    def __init__(self, crop_size, p=0.2):
        self.crop_size = crop_size  # Can be int (square) or tuple (h, w)
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            img_width, img_height = image.size

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
    def __init__(self, crop_size, p=0.2):
        self.p = p
        self.crop_size = crop_size

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask
        else:

            w, h = image.size
            crop_h, crop_w = self.crop_size

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