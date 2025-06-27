import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
import random


class CropDataset(Dataset):
    def __init__(self,
                 image_dir=None,
                 mask_dir=None,
                 non_smoke_dir=None,
                 ijmond_positive_dir=None,
                 ijmond_negative_dir=None,
                 test=False,
                 transform=None,
                 mask_transform=None,
                 img_size=(1024, 1024),
                 backbone='sam',
                 ratio=0.1):

        self.img_size = img_size
        self.samples = []
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.non_smoke_dir = non_smoke_dir
        self.ratio = ratio

        self.ijmond_positive_dir = ijmond_positive_dir
        self.ijmond_negative_dir = ijmond_negative_dir

        # # Load smoke images and masks
        # print("Image Dir:", self.image_dir)
        # print("Mask Dir:", self.mask_dir)

        if self.image_dir and self.mask_dir:
            all_with_mask = []
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, f"mask_{img_name}")

                if os.path.exists(mask_path):
                    if test:
                        self.samples.append({
                            'image': img_path,
                            'mask': mask_path,
                            'label': 1,
                            'is_smoke': True
                        })
                    # print("os.path.exists(mask_path)",os.path.exists(mask_path))
                    else:
                        all_with_mask.append({
                            'image': img_path,
                            'mask': mask_path,
                            'label': 1,
                            'is_smoke': True
                        })

                        num_to_keep = max(1, len(all_with_mask) // 20)
                        sampled_with_mask = random.sample(all_with_mask, num_to_keep)
                        self.samples.extend(sampled_with_mask)

        print("supervised smoke samples:", len(self.samples))

        if self.non_smoke_dir:
            all_non_smoke = [
                os.path.join(self.non_smoke_dir, img_name)
                for img_name in os.listdir(self.non_smoke_dir)
                if not img_name.startswith('.')
            ]

            num_samples = max(1, int(len(all_non_smoke) * self.ratio))

            selected_non_smoke = random.sample(all_non_smoke, num_samples)

            for img_path in selected_non_smoke:
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'label': 0,
                    'is_smoke': False
                })

            # for img_name in sorted(os.listdir(non_smoke_dir)):
            #     if img_name.startswith('.'): continue
            #     img_path = os.path.join(non_smoke_dir, img_name)
            #     self.samples.append({
            #         'image': img_path,
            #         'mask': None,
            #         'label': 0,
            #         'is_smoke': False
            #     })

        print("corped_image", len(self.samples))

        if self.ijmond_positive_dir:
            for img_name in sorted(os.listdir(ijmond_positive_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(ijmond_positive_dir, img_name)
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'label': 1,
                    'is_smoke': True
                })
        print("ijmond positive", len(self.samples))

        if self.ijmond_negative_dir:
            for img_name in sorted(os.listdir(ijmond_negative_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(ijmond_negative_dir, img_name)
                self.samples.append({
                    'image': img_path,
                    'mask': None,
                    'label': 0,
                    'is_smoke': False
                })
        print("ijmond negative", len(self.samples))

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
        img = Image.open(sample['image']).convert('RGB')

        # Load mask if available
        if sample['label'] == 1 and sample['mask']:
            mask = Image.open(sample['mask']).convert('L')  # Grayscale

        elif sample['label'] == 1 and not sample['mask']:
            # mask=None
            mask = Image.new('L', img.size, 0)
        else:
            mask = Image.new('L', img.size, 0)  # Black mask for non-smoke

        img = self.transform(img)
        mask = self.mask_transform(mask)

        return img, torch.tensor(sample['label'], dtype=torch.long), \
        os.path.splitext(os.path.basename(sample['image']))[0], mask


class mix_CropDataset(Dataset):
    def __init__(self,
                 image_dir=None,
                 mask_dir=None,
                 weak=False,
                 label=1,
                 transform=None,
                 mask_transform=None,
                 img_size=(1024, 1024),
                 ):
        self.img_size = img_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.weak = weak
        self.label = label

        self.samples = []
        if self.image_dir:
            for img_name in sorted(os.listdir(image_dir)):
                if img_name.startswith('.'): continue
                img_path = os.path.join(image_dir, img_name)
                if mask_dir:
                    mask_path = os.path.join(mask_dir, f"mask_{img_name}")
                else:
                    mask_path = None
                if weak:
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'label': label,
                        'is_weak': True
                    })
                else:
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'label': label,
                        'is_weak': False
                    })

        print("smoke samples:", len(self.samples))

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
        img = Image.open(sample['image']).convert('RGB')

        # Load mask if available
        if sample['label'] == 1 and sample['mask']:
            mask = Image.open(sample['mask']).convert('L')  # Grayscale
        else:
            mask = Image.new('L', img.size, 0)  # Black mask for non-smoke

        img = self.transform(img)
        mask = self.mask_transform(mask)

        return img, torch.tensor(sample['label'], dtype=torch.long), \
        os.path.splitext(os.path.basename(sample['image']))[0], mask
