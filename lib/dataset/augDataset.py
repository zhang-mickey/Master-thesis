from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
from PIL import Image
from torchvision import transforms
import random
from lib.utils.augmentation import *
import albumentations as A


class augDataset(Dataset):
    def __init__(self,
                 json_path, img_dir, smoke5K_path, Rise_path,
                 transform=None, mask_transform=None,
                 image_ids=None, return_both=False, smoke5k=False, Rise=False):
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # supercategory:smoke category:high-opacity,low-opacity
        self.category_to_supercategory = {category['id']: category['supercategory'] for category in
                                          self.data['categories']}

        self.img_dir = img_dir
        self.transform = transform
        self.mask_transform = mask_transform
        # self.img_info = {img['id']: img for img in self.data['images']}
        self.img_info = image_ids if image_ids is not None else [image['id'] for image in self.data['images']]
        self.annotations = self.data['annotations']
        # self.smoke_dataset=smoke_dataset
        self.image_data = []
        self.image_ids_mapping = {}  # Store {index: image_id}
        self.smoke5k_path = smoke5K_path
        self.smoke5k = smoke5k
        self.Rise = Rise
        self.Rise_path = Rise_path
        self.return_both = return_both

        # load coco dataset
        for image in self.data['images']:
            if image['id'] in self.img_info:
                image_path = os.path.join(img_dir, image['file_name'])
                self.image_data.append(
                    {
                        'path': image_path,
                        'label': 1,
                        'source': 'coco',
                        'mask_path': None
                    })

                self.image_ids_mapping[len(self.image_data) - 1] = f"coco_{image['id']}"
        # load smoke5k dataset
        if self.smoke5k:
            img_dir = os.path.join(smoke5K_path, 'img')
            gt_dir = os.path.join(smoke5K_path, 'gt')

            # Get filtered image/mask lists
            smoke5K_images = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            smoke5k_masks = sorted([
                f for f in os.listdir(gt_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            # Verify file pairs
            assert len(smoke5K_images) == len(smoke5k_masks), "Mismatched image/mask pairs"

            for img_file, mask_file in zip(smoke5K_images, smoke5k_masks):
                img_path = os.path.join(img_dir, img_file)
                mask_path = os.path.join(gt_dir, mask_file)

                self.image_data.append(
                    {
                        'path': img_path,
                        'label': 1,
                        'source': 'smoke5k',
                        'mask_path': mask_path
                    }
                )
                self.image_ids_mapping[len(self.image_data) - 1] = f"smoke5k_{img_file}"

        # load Rise dataset
        if self.Rise:
            rise_images = sorted([
                f for f in os.listdir(Rise_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            for img_file in rise_images:
                img_path = os.path.join(Rise_path, img_file)
                self.image_data.append({
                    'path': img_path,
                    'label': 0,  # 0 for non-smoke
                    'source': 'rise',
                    'mask_path': None
                })
                self.image_ids_mapping[len(self.image_data) - 1] = f"rise_{img_file}"

    def __len__(self):
        # Double length when returning both versions
        return 2 * len(self.image_data) if self.return_both else len(self.image_data)

    def __getitem__(self, idx):
        # Determine if we need to return original or augmented
        is_augmented = False
        if self.return_both:
            original_idx = idx % len(self.image_data)
            is_augmented = idx >= len(self.image_data)
        else:
            original_idx = idx

        item = self.image_data[original_idx]
        source = item['source']

        # Load base image and mask
        image = Image.open(item['path']).convert('RGB')

        # Load mask based on source
        if source == 'coco':
            mask = self._load_coco_mask(original_idx)
        elif source == 'smoke5k':
            mask = np.array(Image.open(item['mask_path']).convert('L')) // 255
        else:  # RISE
            mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Apply augmentations if needed
        if is_augmented:
            image, mask = self._apply_augmentations(image, mask)
            image_id = f"{self.image_ids_mapping[original_idx]}_aug"
        else:
            image_id = self.image_ids_mapping[original_idx]

        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        # Apply additional transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, image_id

    def _load_coco_mask(self, idx):
        img_info = self.data['images'][idx]
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        image_annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        for ann in image_annotations:
            if isinstance(ann['segmentation'], list):
                for poly in ann['segmentation']:
                    poly = np.array(poly).reshape((len(poly) // 2, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
            elif isinstance(ann['segmentation'], dict):
                rle = coco_mask.decode(ann['segmentation'])
                if rle.shape != mask.shape:
                    rle = cv2.resize(rle, (mask.shape[1], mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, rle)
        return mask

    def _apply_augmentations(self, image, mask):
        """Apply strong augmentations"""
        # Convert to numpy for Albumentations
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Define strong augmentations
        aug = A.Compose([
            # A.RandomResizedCrop(512, 512, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            # A.Rotate(limit=30, p=0.5),
            # A.ColorJitter(p=0.5),
            # A.GaussianBlur(p=0.3),
            # A.RandomBrightnessContrast(p=0.5)
        ])

        augmented = aug(image=image_np, mask=mask_np)
        return Image.fromarray(augmented['image']), Image.fromarray(augmented['mask'])