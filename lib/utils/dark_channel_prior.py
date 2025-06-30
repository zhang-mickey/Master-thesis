import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import json
import os

def get_dark_channel(img, size=15):
    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(dc, kernel)
    return dark


def show_dcp_map(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dark_channel = get_dark_channel(image)
    dcp_map = get_dark_channel(image)
    edges = cv2.Canny(image, 100, 200)
    combined_mask = np.maximum(dcp_map, edges.astype(float) / 255.0)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Dark Channel Map")
    plt.imshow(dark_channel, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Dark Channel Map")
    plt.imshow(combined_mask, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# image_dir = "smoke-segmentation.v5i.coco-segmentation/cropped_images/"
image_dir = "frames/positive"
samples = []

for img_name in sorted(os.listdir(image_dir)):
    if img_name.startswith('.'): continue
    img_path = os.path.join(image_dir, img_name)

    samples.append({
                        'image': img_path,
                        'label': 1,
                        'is_smoke': True
                    })
# Choose an image
samples=samples[0]
image_path = samples["image"]
show_dcp_map(image_path)