from pycocotools import mask as coco_mask
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
from PIL import Image
from torchvision import transforms
import os
from albumentations import Compose
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from lib.utils.augmentation import *
def pil_to_cv2(pil_img):
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def split_non_smoke_dataset(non_smoke_folder, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
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
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"Non-smoke images split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    return train_files, val_files, test_files


class SmokeWeaklyDataset(Dataset):
    def __init__(self,
                 annotations_file, images_folder,
                 transform=None, mask_transform=None,
                 image_ids=None,
                 non_smoke_image_folder=None,
                 non_smoke_files=None,
                 smoke_aug=None, smoke_dataset=None,
                 flag=False):

        self.annotations_file = annotations_file
        self.images_folder = images_folder
        self.transform = transform
        self.mask_transform = mask_transform
        self.frames_dir = non_smoke_image_folder

        self.flag = flag
        self.smoke_dataset = smoke_dataset

        # Load COCO annotation file
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.annotations = self.data['annotations']

        if self.smoke_dataset is not None:
            self.smoke_aug = smoke_aug
            # self.smoke_aug = SmokeCopyPaste(self.smoke_dataset, p=0.7)
        else:
            self.smoke_aug = smoke_aug

        self.image_ids = image_ids if image_ids is not None else [image['id'] for image in self.data['images']]
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

        self.image_data = []
        self.image_ids_mapping = {}  # Store {index: image_id}

        for image in self.data['images']:
            if image['id'] in self.image_ids:
                image_path = os.path.join(images_folder, image['file_name'])
                self.image_data.append((image_path, 1))

                self.image_ids_mapping[len(self.image_data) - 1] = f"coco_{image['id']}"

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
        image_ids = self.image_ids_mapping[index]

        if self.flag == True:
            image_np = np.array(image)
            crop_height = int(image_np.shape[0] * 0.05)
            image_cropped = image_np[crop_height:, :, :]
            image = Image.fromarray(image_cropped)

        mask = None  # Initialize mask to None

        if label == 1 and self.smoke_dataset is not None:
            # Look up this image in the smoke dataset
            for i in range(len(self.smoke_dataset)):
                smoke_img, smoke_mask, smoke_id = self.smoke_dataset[i]
                if str(smoke_id) == str(image_ids):
                    smoke_mask_np = smoke_mask.squeeze().numpy()
                    mask = smoke_mask_np[crop_height:, :]  # Crop same way as image
                    break

        # If it's a non-smoke image and you're doing augmentation
        elif label == 0 and self.smoke_dataset is not None:
            # image, label, aug_mask = self.smoke_aug(image, label, return_mask=True)
            if random.random() < 0.5:
                # 1. Randomly pick a smoke image + mask
                rand_index = np.random.randint(0, len(self.smoke_dataset))
                smoke_img, smoke_mask, smoke_id = self.smoke_dataset[rand_index]

                # Convert PIL to numpy
                non_smoke_np = np.array(image)
                smoke_np = np.array(smoke_img)

                print("shape of ",non_smoke_np.shape) # (570, 600, 3)
                print("shape of smoke_img",smoke_img.shape) #torch.Size([3, 512, 512])

                mask_np = smoke_mask.squeeze().numpy()

                # Resize smoke and mask to fit the non-smoke image if needed
                if smoke_np.shape != non_smoke_np.shape:
                    if len(non_smoke_np.shape) == 3 and non_smoke_np.shape[2] == 3:
                        # Convert from (C, H, W) to (H, W, C)
                        smoke_np = np.transpose(smoke_np, (1, 2, 0))
                    smoke_np = cv2.resize(smoke_np, (non_smoke_np.shape[1], non_smoke_np.shape[0]))
                    smoke_np = (smoke_np * 255).astype(np.uint8)
                    mask_np = cv2.resize(mask_np, (non_smoke_np.shape[1], non_smoke_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # 2. Copy-paste using mask
                pasted_image = non_smoke_np.copy()
                mask_bool = mask_np > 0  # shape (H, W)

                # Apply the mask to each RGB channel
                for c in range(3):  # RGB channels
                    pasted_image[:, :, c][mask_bool] = smoke_np[:, :, c][mask_bool]
                # 3. New mask
                mask = (mask_np > 0).astype(np.uint8) * 255
                # 4. Convert back to PIL

                # 4. Convert back to PIL for transforms
                image = Image.fromarray(pasted_image.astype(np.uint8))
                # mask = Image.fromarray(mask.astype(np.uint8))

                # Optional: change label to 1 since now it has smoke
                label = 1

            else:
                mask = np.zeros(image.size[::-1], dtype=np.uint8)  # Empty mask for non-smoke

        if isinstance(mask, np.ndarray):
            # Check if mask is 1D and reshape it if needed
            if len(mask.shape) == 1:
                # Reshape 1D mask to 2D (assuming it's a square)
                side_length = int(np.sqrt(mask.shape[0]))
                mask = mask.reshape(side_length, side_length)

        elif isinstance(mask, torch.Tensor):
            # If it's already a tensor, ensure it has channel dimension
            if len(mask.shape) == 1:
                # Reshape 1D tensor to 2D
                side_length = int(np.sqrt(mask.shape[0]))
                mask = mask.reshape(1, side_length, side_length)  # Add channel dimension

        if mask is not None:
            if isinstance(mask, np.ndarray):
                # For numpy arrays, use astype
                mask = mask.astype(np.uint8)
            elif isinstance(mask, torch.Tensor):
                # For PyTorch tensors, use to() method instead of astype
                mask = mask.cpu().numpy().astype(np.uint8)
            mask = Image.fromarray(mask)  # mask should now be uint8 and a 2D array
        else:
            if isinstance(image, Image.Image):
                mask = Image.new('L', image.size, 0)
            else:
                # If image is already transformed, create a default mask
                mask = Image.new('L', (512, 512), 0)  # Default size, adjust as needed

        if self.transform:
            image = self.transform(image)
        if self.mask_transform and mask is not None:
            mask = self.mask_transform(mask)

        return image, torch.tensor(label, dtype=torch.float), image_ids, mask



