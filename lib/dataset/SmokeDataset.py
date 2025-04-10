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
import matplotlib.pyplot as plt

from lib.utils.augmentation import *


class SmokeDataset(Dataset):
    def __init__(self,
                 json_path, img_dir, smoke5K_path, Rise_path,
                 transform=None, mask_transform=None,
                 image_ids=None, return_both=False,
                 smoke5k=False, Rise=False):
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # supercategory:smoke category:high-opacity,low-opacity
        self.category_to_supercategory = {category['id']: category['supercategory'] for category in
                                          self.data['categories']}

        self.img_dir = img_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_info = image_ids if image_ids is not None else [image['id'] for image in self.data['images']]
        self.annotations = self.data['annotations']
        self.flag = return_both
        self.image_data = []
        self.image_ids_mapping = {}  # Store {index: image_id}

        self.smoke5k_path = smoke5K_path
        self.smoke5k = smoke5k
        self.Rise = Rise
        self.Rise_path = Rise_path

        # load coco dataset
        for image in self.data['images']:
            if image['id'] in self.img_info:
                image_path = os.path.join(img_dir, image['file_name'])
                self.image_data.append(
                    {
                        'path': image_path,
                        'label': 1,
                        'source': 'coco',
                        'mask_path': None,
                        'image_id': image['id']
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
                    'label': 0,
                    'source': 'rise',
                    'mask_path': None
                })
                self.image_ids_mapping[len(self.image_data) - 1] = f"rise_{img_file}"

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):

        item = self.image_data[idx]
        source = item['source']
        label = item['label']
        # loading for both datasets
        image = Image.open(item['path']).convert('RGB')

        if source == 'coco':
            coco_id = item['image_id']
            mask = self._load_coco_mask(coco_id)

            image_np = np.array(image)
            crop_height = int(image_np.shape[0] * 0.1)
            image_cropped = image_np[crop_height:, :, :]
            crop_left = int(image_np.shape[0] * 0.22)
            image_cropped = image_cropped[:,crop_left :, :]
            image = Image.fromarray(image_cropped)

            mask_np = np.array(mask)
            crop_height = int(mask_np.shape[0] * 0.1)
            mask_cropped = mask_np[crop_height:, :]
            crop_left = int(mask_np.shape[0] * 0.22)
            mask_cropped = mask_cropped[:,crop_left :]
            mask = Image.fromarray(mask_cropped)

            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.title('Image')
            # plt.subplot(1, 2, 2)

            # plt.imshow(mask)
            # plt.title('Mask')
            # plt.show()
        elif source == 'rise':
            # zero_mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            # Smoke5K processing
            mask = Image.open(item['mask_path']).convert('L')
            mask = np.array(mask) // 255  # Convert to binary mask
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)


        if self.flag == True:
            image, mask = self._apply_augmentations(image, mask)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.flag == False:
            return image, mask, self.image_ids_mapping[idx], label

        # print("image/mask",image.shape, mask.shape)

        return image, mask, self.image_ids_mapping[idx], label

    def _load_coco_mask(self, idx):
        img_info = next(img for img in self.data['images'] if img['id'] == idx)

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
        """Apply identical augmentations to image and mask"""
        # Random cropping
        if random.random() < 0.2:
            # original 1920*1080
            image, mask = RandomCrop_For_Segmentation(crop_size=(1720, 900))(image, mask)
        else:
            image, mask = CenterCrop_For_Segmentation(crop_size=(1720, 900))(image, mask)

        # Color jitter (only on images)
        image = ColorJitterTransform(brightness=0.2, contrast=0.2,
                                     saturation=0.2, hue=0.1, p=0.5)(image)

        # Random horizontal flip
        # if random.random() < 0.5:
        #     image = transforms.functional.hflip(image)
        #     mask = transforms.functional.hflip(mask)

        return image, mask