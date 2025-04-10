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

import matplotlib.pyplot as plt


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
                 transform=None,
                 mask_transform=None,
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
                self.image_data.append((
                    image_path,
                    1))

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
        # image = Image.open(image_path).convert("RGB")
        image_ids = self.image_ids_mapping[index]

        # if self.flag == True:

        mask = None  # Initialize mask to None

        if label == 1 and self.smoke_dataset is not None:
            # Look up this image in the smoke dataset
            for i in range(len(self.smoke_dataset)):
                smoke_image, smoke_mask, smoke_id, labels = self.smoke_dataset[i]
                if str(smoke_id) == str(image_ids):
                    image = smoke_image
                    mask = smoke_mask
                    # mask= smoke_mask.squeeze().numpy()

                    # image_np = np.array(smoke_img)
                    # crop_height = int(image_np.shape[0] * 0.1)
                    # image_cropped = image_np[crop_height:, :, :]
                    # crop_left = int(image_np.shape[0] * 0.22)
                    # image_cropped = image_cropped[:,crop_left :, :]
                    # image = Image.fromarray(image_cropped)

                    # mask_np = np.array(mask)
                    # crop_height = int(mask_np.shape[0] * 0.1)
                    # mask_cropped = mask_np[crop_height:, :]
                    # crop_left = int(mask_np.shape[0] * 0.22)
                    # mask_cropped = mask_cropped[:,crop_left :]
                    # mask = Image.fromarray(mask_cropped)
                    image = np.transpose(image, (1, 2, 0))

                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.title('Image')
                    plt.subplot(1, 2, 2)

                    plt.imshow(mask)
                    plt.title('Mask')
                    plt.show()
                    image = np.transpose(image, (2, 0, 1))
                    break

        # If it's a non-smoke image and you're doing augmentation
        elif label == 0 and self.smoke_dataset is not None:
            # image, label, aug_mask = self.smoke_aug(image, label, return_mask=True)
            image = Image.open(image_path).convert("RGB")
            if random.random() < 0.5:
                # 1. Randomly pick a smoke image + mask
                rand_index = np.random.randint(0, len(self.smoke_dataset))
                smoke_img, smoke_mask, smoke_id, labels = self.smoke_dataset[rand_index]

                # Convert PIL to numpy
                non_smoke_np = np.array(image)
                smoke_np = np.array(smoke_img)

                # smoke_np = np.transpose(smoke_np, (1, 2, 0))
                # plt.imshow(smoke_np)
                # plt.show()

                # print("shape of ",non_smoke_np.shape) # (570, 600, 3)
                # print("shape of smoke_img",smoke_img.shape) #torch.Size([3, 512, 512])

                mask_np = smoke_mask.squeeze().numpy()

                # Resize smoke and mask to fit the non-smoke image if needed
                if smoke_np.shape != non_smoke_np.shape:
                    if len(non_smoke_np.shape) == 3 and non_smoke_np.shape[2] == 3:
                        # Convert from (C, H, W) to (H, W, C)
                        smoke_np = np.transpose(smoke_np, (1, 2, 0))
                    smoke_np = (smoke_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    smoke_np = np.clip(smoke_np, 0, 255).astype(np.uint8)

                    smoke_np = cv2.resize(smoke_np, (non_smoke_np.shape[1], non_smoke_np.shape[0]))

                    # smoke_np = (smoke_np * 255).astype(np.uint8)

                    # plt.imshow(smoke_np)
                    # plt.show()
                    #
                    # mean = [0.485, 0.456, 0.406]
                    # std = [0.229, 0.224, 0.225]
                    # smoke_np = denormalize( smoke_np, mean, std)

                    mask_np = cv2.resize(mask_np, (non_smoke_np.shape[1], non_smoke_np.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    # 2. Copy-paste using mask
                pasted_image = non_smoke_np.copy()

                # Apply the mask to each RGB channel

                # print("shape of ",pasted_image.shape)
                # print("shape of smoke_img",smoke_np.shape)
                # pasted_image=pil_to_cv2(pasted_image)
                # smoke_np=pil_to_cv2(smoke_np)
                # for c in range(3):  # RGB channels
                #     pasted_image[:, :, c][mask_bool] = smoke_np[:, :, c][mask_bool]
                # smoke_np = smoke_np.astype(np.uint8)
                # pasted_image = pasted_image.astype(np.uint8)
                # Create mask with 3 channels
                mask_bool_3d = (mask_np > 0)[..., None]  # Shape: (H, W, 1)
                mask_bool_3d = np.repeat(mask_bool_3d, 3, axis=2)  # Now shape: (H, W, 3)

                # Apply the mask
                # print("smoke_np.dtype",smoke_np.dtype)
                # print("pasted_image.dtype",pasted_image.dtype)

                pasted_image = np.where(mask_bool_3d, smoke_np, pasted_image)

                mask = (mask_np > 0).astype(np.uint8) * 255
                # 4. Convert back to PIL for transforms
                image = Image.fromarray(pasted_image.astype(np.uint8))

                # mask = Image.fromarray(mask.astype(np.uint8))

                # plt.figure(figsize=(12, 6))
                # plt.subplot(1, 3, 1)
                #
                # plt.imshow(image)
                #
                # plt.subplot(1, 3, 2)
                # plt.imshow(smoke_np)
                #
                # plt.subplot(1, 3, 3)
                # plt.imshow(mask_np)
                #
                # plt.show()
                label = 1

            else:
                mask = np.zeros(image.size[::-1], dtype=np.uint8)  # Empty mask for non-smoke

        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 1:
                side_length = int(np.sqrt(mask.shape[0]))
                mask = mask.reshape(side_length, side_length)

        elif isinstance(mask, torch.Tensor):
            if len(mask.shape) == 1:
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



