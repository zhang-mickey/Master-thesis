
from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from PIL import Image
from torchvision import transforms

class SmokeDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, mask_transform=None):
        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_info = {img['id']: img for img in self.data['images']}
        self.annotations = self.data['annotations']

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        # Load image
        img_info = list(self.img_info.values())[idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create an empty mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Load annotations
        image_annotations = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        for ann in image_annotations:
            if isinstance(ann['segmentation'], list):
                for poly in ann['segmentation']:
                    poly = np.array(poly).reshape((len(poly) // 2, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)  # Set mask pixels to 1

            elif isinstance(ann['segmentation'], dict):  # RLE encoding
                rle = coco_mask.decode(ann['segmentation'])
                if rle.shape != mask.shape:
                    print(rle.shape)
                    print(mask.shape)
                    rle = cv2.resize(rle, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                mask = np.maximum(mask, rle)

        # Convert to PIL Image before transformation
        # image = transforms.ToPILImage()(image)
        image = Image.fromarray(image)
        mask = transforms.ToPILImage()(mask * 255)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask