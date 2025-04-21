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


class PseudocamDataset(Dataset):
    def __init__(self,
                 image_dir,
                 cam_dir,
                 mask_dir=None,
                 transform=None,
                 mask_transform=None,
                 image_ids=None,
                 affinity=None):
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.mask_dir = mask_dir

        self.samples = []

        if self.image_dir and self.cam_dir:
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.startswith('.'):
                    continue

                img_path = os.path.join(image_dir, img_name)
                cam_path = os.path.join(cam_dir, f"cam_{img_name}")

                mask_path = os.path.join(mask_dir, f"mask_{img_name}") if mask_dir else None

                if os.path.exists(cam_path):
                    self.samples.append({
                        'image': img_path,
                        'cam': cam_path,
                        'mask': mask_path if mask_path and os.path.exists(mask_path) else None,
                        'label': 1,
                        'is_smoke': True
                    })
        print("len(self.samples)", len(self.samples))
        print("mask_path", mask_path)
        print("cam_path", cam_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image']).convert('RGB')
        cam = Image.open(sample['cam']).convert('L') if sample['is_smoke'] else Image.new('L', img.size, 0)

        cam_np = np.array(cam)
        print(f"CAM min: {cam_np.min()}, max: {cam_np.max()}, mean: {cam_np.mean()}")

        # Load real mask if available
        if sample.get('mask'):
            mask = Image.open(sample['mask']).convert('L')
        else:
            mask = Image.new('L', img.size, 0)

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            # cam = self.mask_transform(cam)
            mask = self.mask_transform(mask)

        to_tensor = transforms.ToTensor()  # Converts to float tensor in [0,1]
        cam = to_tensor(cam)

        if isinstance(cam, torch.Tensor):
            print(f"CAM after transform - min: {cam.min().item()}, max: {cam.max().item()}, mean: {cam.mean().item()}")

        print("cam", cam.shape)
        return (
            img,
            torch.tensor(sample['label'], dtype=torch.long),
            os.path.splitext(os.path.basename(sample['image']))[0],  # image name
            cam,
            mask
        )

