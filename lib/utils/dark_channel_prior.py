import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
import json
import os


def apply_purple_overlay(image, dcp_map, threshold=0.4, alpha=0.5):
    """
    Applies a purple overlay on the image where DCP values > threshold.

    Args:
        image: Original image (np.uint8, RGB)
        dcp_map: Normalized float array (0 to 1)
        threshold: Value to highlight (e.g., smoke regions)
        alpha: Transparency factor of overlay (0 to 1)

    Returns:
        Combined image with purple overlay
    """
    # Threshold DCP to binary mask
    mask = (dcp_map > threshold).astype(np.uint8) * 255

    # Create empty purple mask
    purple_mask = np.zeros_like(image)
    purple_mask[:, :, 0] = 255  # R
    purple_mask[:, :, 1] = 0  # G
    purple_mask[:, :, 2] = 0  # B

    # Apply mask via bitwise_and
    colored_mask = cv2.bitwise_and(purple_mask, purple_mask, mask=mask)

    # Blend with original image
    result = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

    return result, colored_mask


def visualize_dcp_purple(image_path, threshold=0.99, alpha=0.7):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Compute DCP
    dcp_map = get_dark_channel(image)

    # Apply purple overlay
    overlay_image, _ = apply_purple_overlay(image, dcp_map, threshold, alpha)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Purple Highlighted Regions")
    plt.imshow(overlay_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_dark_channel(img, size=15):
    """
    Compute the dark channel of an image.

    Args:
        image: Input image (np.array, H x W x 3, np.uint8)
        patch_size: Size of local patch used for minima extraction

    Returns:
        dark_channel: Computed dark channel (H x W)
    """

    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(dc, kernel)
    return dark


def show_dcp_map(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Compute dark channel
    dark_channel = get_dark_channel(image)
    dcp_map = get_dark_channel(image)
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Combine DCP and edges
    combined_mask = np.maximum(dcp_map, edges.astype(float) / 255.0)

    # Plotting
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


image_dir = "smoke-segmentation.v5i.coco-segmentation/cropped_images/"
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
samples=samples[121]
image_path = samples["image"]
# visualize_dcp_purple(image_path)
show_dcp_map(image_path)