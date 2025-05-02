import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image
from torch.distributions.beta import Beta


class LR_CropDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 non_smoke_dir=None,

                 transform=None,
                 mask_transform=None,
                 img_size=(512, 512)):

        self.img_size = img_size
        self.samples = []
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.non_smoke_dir = non_smoke_dir

        if self.image_dir and self.mask_dir:
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, f"mask_{img_name}")
                if os.path.exists(mask_path):
                    # print("os.path.exists(mask_path)",os.path.exists(mask_path))
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'label': 1,
                        'is_smoke': True
                    })
        print("len(self.samples)", len(self.samples))

        # Load non-smoke images

        if self.non_smoke_dir:
            for img_name in sorted(os.listdir(non_smoke_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(non_smoke_dir, img_name)
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'label': 0,
                    'is_smoke': False
                })

        print("len(self.samples)", len(self.samples))

        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomErasing(p=0.1,
                                     # try to aviod erasing smoke regions
                                     scale=(0.02, 0.1),
                                     ratio=(0.3, 1),
                                     # random fulfill
                                     value='random',
                                     inplace=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.RandomErasing(p=0.1,
                                     # try to aviod erasing smoke regions
                                     scale=(0.02, 0.1),
                                     ratio=(0.3, 1),
                                     # random fulfill
                                     value='random',
                                     inplace=False),
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

        return img, torch.tensor(sample['label'], dtype=torch.float).unsqueeze(0)
