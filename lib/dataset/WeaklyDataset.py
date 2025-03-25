
from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from PIL import Image
from torchvision import transforms
import os 

def split_non_smoke_dataset(non_smoke_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split non-smoke images into train, validation, and test sets.
    
    Args:
        non_smoke_folder: Path to the folder containing non-smoke images
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        test_ratio: Ratio of images for testing
        
    Returns:
        train_files, val_files, test_files: Lists of image filenames for each set
    """
    if not os.path.exists(non_smoke_folder):
        print(f"Warning: Non-smoke folder {non_smoke_folder} does not exist")
        return [], [], []
        
    # Get all image files
    image_files = [f for f in os.listdir(non_smoke_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No image files found in {non_smoke_folder}")
        return [], [], []
    
    # Shuffle the files
    np.random.shuffle(image_files)
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Split the dataset
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    print(f"Non-smoke images split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return train_files, val_files, test_files

class SmokeWeaklyDataset(Dataset):
    def __init__(self, annotations_file, images_folder, transform=None,image_ids=None,non_smoke_image_folder=None,non_smoke_files=None):
        self.annotations_file = annotations_file
        self.images_folder = images_folder
        self.transform = transform
        self.frames_dir = non_smoke_image_folder

        # Load COCO annotation file
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

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

        # self.image_ids = list(self.image_annotations.keys())

        # Load image paths and labels
        self.image_data=[]
        self.image_ids_mapping = {}  # Store {index: image_id}

        for image in self.data['images']:
            if image['id'] in self.image_ids:
                image_path = os.path.join(images_folder, image['file_name'])
                self.image_data.append((image_path, self.image_labels[image['id']]))
               
                # Ensure data is added before accessing the last index
                self.image_ids_mapping[len(self.image_data) - 1] = str(image['id'])  # Use COCO image ID

        if non_smoke_image_folder is not None and non_smoke_files is not None:
            for filename in non_smoke_files:
                if filename.lower().endswith(('jpg', 'jpeg', 'png')):  # Ensure it's an image file
                    image_path = os.path.join(non_smoke_image_folder, filename)
                    self.image_data.append((image_path, 0))  # Assign label 0 (non-smoke)

                    if len(self.image_data) > 0:  # Ensure non-empty list before assignment
                        image_id = os.path.splitext(filename)[0]  
                        self.image_ids_mapping[len(self.image_data) - 1] = image_id
    
        print(f"Total images loaded: {len(self.image_data)}")
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        # Load image and label
        image_path, label = self.image_data[index]
        image = Image.open(image_path).convert("RGB")

        image_ids=self.image_ids_mapping[index]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float),image_ids



        