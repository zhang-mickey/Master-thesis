from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import os
import sys
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random

project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
sys.path.append(project_root)


def crop_logo(image_np, mask_np, crop_height_ratio=0.1, crop_left_ratio=0.22):
    """Crop logo from both image and mask using numpy arrays"""
    assert image_np.shape[:2] == mask_np.shape[:2], "Image and mask dimensions mismatch"

    h, w = image_np.shape[:2]
    crop_height = int(h * crop_height_ratio)
    crop_left = int(w * crop_left_ratio)

    img_cropped = image_np[crop_height:, crop_left:, :]
    mask_cropped = mask_np[crop_height:, crop_left:]
    return img_cropped, mask_cropped


def crop_images_and_masks(json_path, image_folder, output_image_folder, output_mask_folder,
                          output_non_smoke_image_folder=None, crop_size=512, stride=256,
                          non_smoke_ratio=1, max_non_smoke_per_image=5):
    """
    Main cropping function with smoke/non-smoke balance
    """
    total_smoke = 0
    total_non_smoke = 0

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    if output_non_smoke_image_folder:
        os.makedirs(output_non_smoke_image_folder, exist_ok=True)

    coco = COCO(json_path)
    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info['file_name'])
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Create original mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            ann_mask = coco.annToMask(ann)
            mask = np.where(ann_mask > 0, 1, mask)

        try:
            img_cropped, mask_cropped = crop_logo(img, mask)
        except AssertionError:
            continue

        h_new, w_new = img_cropped.shape[:2]
        if h_new < crop_size or w_new < crop_size:
            continue

        base_name = os.path.splitext(img_info['file_name'])[0]
        smoke_positions = []
        non_smoke_positions = []
        smoke_count = 0

        # First pass: collect positions and save smoke crops
        for y in range(0, h_new - crop_size + 1, stride):
            for x in range(0, w_new - crop_size + 1, stride):
                crop_img = img_cropped[y:y + crop_size, x:x + crop_size]
                crop_mask = mask_cropped[y:y + crop_size, x:x + crop_size]

                if np.any(crop_mask):
                    cv2.imwrite(os.path.join(output_image_folder, f"{base_name}_smoke_{x}_{y}.png"), crop_img)
                    cv2.imwrite(os.path.join(output_mask_folder, f"mask_{base_name}_{x}_{y}.png"), crop_mask * 255)
                    smoke_positions.append((x, y))
                    smoke_count += 1
                    total_smoke += 1
                else:
                    non_smoke_positions.append((x, y))

        # Second pass: save non-smoke crops
        if output_non_smoke_image_folder and non_smoke_positions:
            num_non_smoke = min(
                int(smoke_count * non_smoke_ratio),
                max_non_smoke_per_image,
                len(non_smoke_positions)
            )
            print("num_non_smoke",num_non_smoke)
            if num_non_smoke > 0:
                random.shuffle(non_smoke_positions)
                for x, y in non_smoke_positions[:num_non_smoke]:
                    crop_img = img_cropped[y:y + crop_size, x:x + crop_size]
                    cv2.imwrite(os.path.join(output_non_smoke_image_folder, f"{base_name}_nonsmoke_{x}_{y}.png"),
                                crop_img)
                    total_non_smoke += 1

    print("\n" + "=" * 50)
    print(f"SMOKE CROPS: {total_smoke}")
    print(f"NON-SMOKE CROPS: {total_non_smoke}")
    print(f"TOTAL CROPS: {total_smoke + total_non_smoke}")
    print("=" * 50)


def visualize_cropping(image_path, crop_size=512, stride=256):
    """Visualize cropping process for a single image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create dummy mask for visualization
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Apply logo crop
    img_cropped, mask_cropped = crop_logo(img, mask)

    # Generate grid
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # Original image
    ax[0].imshow(img)
    ax[0].set_title("Original Image")

    # After logo crop
    ax[1].imshow(img_cropped)
    ax[1].set_title("After Logo Removal")

    # Crop positions
    ax[2].imshow(img_cropped)
    h, w = img_cropped.shape[:2]
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            rect = plt.Rectangle((x, y), crop_size, crop_size,
                                 linewidth=1, edgecolor='lime', facecolor='none')
            ax[2].add_patch(rect)
    ax[2].set_title(f"Crop Grid ({crop_size}x{crop_size}, stride {stride})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example cropping
    crop_images_and_masks(
        json_path=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
        image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
        output_image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images"),
        output_mask_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks"),
        output_non_smoke_image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/non_smoke_images"),
        crop_size=512,
        stride=256,
        non_smoke_ratio=0.5,
        max_non_smoke_per_image=8
    )

    # Example visualization
    sample_image = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/kooks_2__2024-11-15T10-49-07Z_frame_2333_jpg.rf.f7862d504c0ac32843fc5a21a819f14b.jpg")
    if os.path.exists(sample_image):
        visualize_cropping(sample_image)
    else:
        print(f"Sample image not found at {sample_image}")