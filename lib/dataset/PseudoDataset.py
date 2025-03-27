# After imports, add this class

from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from PIL import Image
from torchvision import transforms
import os 


class PseudoLabelDataset(Dataset):
    def __init__(self, json_path, image_folder, pseudo_labels_path, 
                 transform=None, mask_transform=None, image_ids=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_folder = image_folder
        self.pseudo_labels_path = pseudo_labels_path

        # Load COCO data
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create image ID to info mapping
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}

        # Filter images by provided IDs
        if image_ids is not None:
            valid_ids = [img_id for img_id in image_ids if img_id in self.image_id_to_info]
            self.image_ids = valid_ids
        else:
            self.image_ids = list(self.image_id_to_info.keys())

        # Verify existence of pseudo labels
        self.missing_masks = []
        for img_id in self.image_ids:
            mask_path = os.path.join(self.pseudo_labels_path, f"pseudo_label_{img_id}.png")
            if not os.path.exists(mask_path):
                self.missing_masks.append(img_id)

        if self.missing_masks:
            print(f"Warning: {len(self.missing_masks)} images missing pseudo masks")

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        # Load image
        img_path = os.path.join(self.image_folder, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Load pseudo-label mask
        mask_path = os.path.join(self.pseudo_labels_path, f"pseudo_label_{image_id}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create empty mask with original image dimensions
            mask = Image.new('L', image.size, 0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask