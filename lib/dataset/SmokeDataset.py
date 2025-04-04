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
from lib.utils.augmentation import RandomCrop_For_Segmentation, CenterCrop_For_Segmentation, ColorJitterTransform


class SmokeDataset(Dataset):
    def __init__(self, json_path, img_dir, smoke5K_path,
                 transform=None, mask_transform=None,
                 image_ids=None, flag=1, smoke5k=False):
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
        self.flag = flag
        # self.smoke_dataset=smoke_dataset
        self.image_data = []
        self.image_ids_mapping = {}  # Store {index: image_id}
        self.smoke5k_path = smoke5K_path
        self.smoke5k = smoke5k

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

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        source = item['source']
        # loading for both datasets
        image = Image.open(item['path']).convert('RGB')

        if source == 'coco':
            img_info = self.data['images'][idx]
            mask = self._load_coco_mask(idx)
        else:
            # Smoke5K processing
            mask = Image.open(item['mask_path']).convert('L')
            mask = np.array(mask) // 255  # Convert to binary mask

        if self.flag == 1:

            probability = random.random()

            if probability < 0.2:
                randomcrop = RandomCrop_For_Segmentation(468)
                image, mask = randomcrop(image, Image.fromarray(mask))
            else:
                crop_transform = CenterCrop_For_Segmentation(468)
                image, mask = crop_transform(image, Image.fromarray(mask))

            image = ColorJitterTransform(brightness=0.2, contrast=0.2,
                                         saturation=0.2, hue=0.1, p=0.5)(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.flag == 2:
            return image, mask, self.image_ids_mapping[idx]

        return image, mask, self.image_ids_mapping[idx]

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