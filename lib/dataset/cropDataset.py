import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image


class CropDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir=None,
                 non_smoke_dir=None,
                 transform=None,
                 mask_transform=None,
                 img_size=(512, 512)):
        """
        Args:
            image_dir: Path to directory with smoke images
            mask_dir: Path to directory with masks
            non_smoke_dir: Path to directory with non-smoke images
            transform: Transformations for images
            mask_transform: Transformations for masks
            img_size: Target image size (H, W)
        """
        self.img_size = img_size
        self.samples = []

        # Load smoke images and masks
        if image_dir and mask_dir:
            for img_name in os.listdir(image_dir):
                if img_name.startswith('.'): continue
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, f"mask_{img_name}")
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'label': 1,
                        'is_smoke': True
                    })

        # Load non-smoke images
        if non_smoke_dir:
            for img_name in os.listdir(non_smoke_dir):
                if img_name.startswith('.'): continue
                img_path = os.path.join(non_smoke_dir, img_name)
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'label': 0,
                    'is_smoke': False
                })

        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample['image']).convert('RGB')

        # Load mask if available
        if sample['is_smoke']:
            mask = Image.open(sample['mask']).convert('L')  # Grayscale
        else:
            mask = Image.new('L', img.size, 0)  # Black mask for non-smoke

        if self.transform and sample['is_smoke']:

            img = self.transform(img)
            mask = self.mask_transform(mask)
        else:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return img, torch.tensor(sample['label'], dtype=torch.long), \
        os.path.splitext(os.path.basename(sample['image']))[0], mask

# if __name__ == "__main__":
#     # Define transformations with augmentation
#     train_transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     mask_transform = transforms.Compose([
#         transforms.Resize((512, 512), interpolation=Image.NEAREST),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor()
#     ])