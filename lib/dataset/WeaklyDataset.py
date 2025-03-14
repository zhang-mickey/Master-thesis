
from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from PIL import Image
from torchvision import transforms


class SmokeWeaklyDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transform=None,image_ids=None):
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        self.images_folder = images_folder
        self.transform = transform
        self.image_ids = image_ids if image_ids is not None else [image['id'] for image in self.data['images']]
        # self.image_annotations = {image['id']: [] for image in self.data['images']}
        # self.image_annotations = {image['id']: [] for image in self.data['images'] if image['id'] in self.image_ids}

        self.image_annotations = {image_id: [] for image_id in self.image_ids}

        for annotation in self.data['annotations']:
            if annotation['image_id'] in self.image_annotations:
                self.image_annotations[annotation['image_id']].append(annotation['category_id'])

        # Map category_id to supercategory
        self.category_to_supercategory = {category['id']: category['supercategory'] for category in
                                          self.data['categories']}
        # Convert to binary labels (smoke vs non-smoke)
        self.image_labels = {}

        for image_id, annotations in self.image_annotations.items():
            # For each image, determine if it belongs to "smoke" or "non-smoke"
            label = 0  # default is non-smoke
            for category_id in annotations:
                supercategory = self.category_to_supercategory[category_id]
                if supercategory == "smoke":
                    label = 1  # If any category is of "smoke", label as smoke
                    break
            self.image_labels[image_id] = label

        self.image_ids = list(self.image_annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image_id for the current sample
        image_id = self.image_ids[index]
        label = self.image_labels[image_id]

        # image_path = f"{self.images_folder}/{self.data['images'][image_id]['file_name']}"
        image_info = next(img for img in self.data['images'] if img['id'] == image_id)
        image_path = f"{self.images_folder}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)