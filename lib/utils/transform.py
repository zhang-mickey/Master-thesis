import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

def crop_logo(image,mask,crop_height_ratio=0.1,crop_left_ratio=0.22):

    image_np = np.array(image)

    crop_height = int(image_np.shape[0] * crop_height_ratio)
    image_cropped = image_np[crop_height:, :, :]
    crop_left = int(image_np.shape[0] * crop_left_ratio)
    image_cropped = image_cropped[:,crop_left :, :]
    image = Image.fromarray(image_cropped)

    mask_np = np.array(mask)
    crop_height = int(mask_np.shape[0] * crop_height_ratio)
    mask_cropped = mask_np[crop_height:, :]
    crop_left = int(mask_np.shape[0] * crop_left_ratio)
    mask_cropped = mask_cropped[:,crop_left :]
    mask = Image.fromarray(mask_cropped)

    return image,mask



def HWC_to_CHW(tensor, sal=False):
    if sal:
        tensor = np.expand_dims(tensor, axis=0)
    else:
        tensor = np.transpose(tensor, (2, 0, 1))
    return tensor


class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img
#define transforms to preprocess input image into format expected by model

def get_transforms(img_size=512):
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Convert PIL to tensor
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),

        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0)))  # Convert to binary mask
        transforms.Lambda(lambda x: (x > 0).float().squeeze(0))
    ])

    return image_transform, mask_transform

def reverse_transforms():
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    return inv_normalize