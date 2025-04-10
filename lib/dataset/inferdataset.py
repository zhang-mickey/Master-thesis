from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
import torchvision
from PIL import Image
from torchvision import transforms
import random
import matplotlib.pyplot as plt

from lib.utils.augmentation import *


class InferDataset(Dataset):
    def __init__(self,
                 json_path, img_dir,
                 transform=torchvision.transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                    )
                ,
                mask_transform=None,
                 image_ids=None,
                ):
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
        self.image_data = []
        self.image_ids_mapping = {}  # Store {index: image_id}


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

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_list=[]

        item = self.image_data[idx]
        source = item['source']
        label = item['label']
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

        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        for s in[256,512,768]:
            img = transforms.Resize((s, s))(image)
            img_list.append(img)


        if self.transform:
            img_list = [self.transform(img) for img in img_list]


        aug_img_list=[]

        for i in range(len(img_list)):
            aug_img_list.append(image)
            aug_img_list.append(np.flip(image,-1).copy)


        if self.mask_transform:
            mask = self.mask_transform(mask)

        return aug_img_list, mask, self.image_ids_mapping[idx], label



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