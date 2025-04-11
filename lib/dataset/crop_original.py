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


def crop_logo(image_np, mask_np, crop_height_ratio=0.12, crop_left_ratio=0.24):
    """Crop logo from both image and mask using numpy arrays"""
    assert image_np.shape[:2] == mask_np.shape[:2], "Image and mask dimensions mismatch"

    h, w = image_np.shape[:2]
    crop_height = int(h * crop_height_ratio)
    crop_left = int(w * crop_left_ratio)

    img_cropped = image_np[crop_height:, crop_left:, :]
    mask_cropped = mask_np[crop_height:, crop_left:]
    return img_cropped, mask_cropped


def crop_images_and_masks(json_path, image_folder, output_image_folder, output_mask_folder,
                          output_non_smoke_image_folder=None, crop_size=512,
                          stride=64,
                          non_smoke_ratio=1,
                          max_non_smoke_per_image=2,
                          min_mask_area_ratio=0.005,
                          top_smoke_regions=3
                          ):

    total_smoke = 0
    total_non_smoke = 0

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    if output_non_smoke_image_folder:
        os.makedirs(output_non_smoke_image_folder, exist_ok=True)

    coco = COCO(json_path)
    image_ids = coco.getImgIds()
    print("length of images",len(image_ids))

    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info['file_name'])
        img = cv2.imread(img_path)

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

        viz_img = img_cropped.copy()
        viz_img = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
        overlay = viz_img.copy()
        alpha = 0.4  # Transparency factor

        h_new, w_new = img_cropped.shape[:2]
        if h_new < crop_size or w_new < crop_size:
            continue

        base_name = os.path.splitext(img_info['file_name'])[0]
        smoke_positions = []
        non_smoke_positions = []
        smoke_count = 0

        smoke_candidates = []

        #  save smoke crops
        for y in range(0, h_new - crop_size + 1, stride):
            for x in range(0, w_new - crop_size + 1, stride):

                crop_img = img_cropped[y:y + crop_size, x:x + crop_size]
                crop_mask = mask_cropped[y:y + crop_size, x:x + crop_size]

                mask_area=np.sum(crop_mask)
                crop_area = crop_size * crop_size
                mask_ratio = mask_area / crop_area

                if mask_ratio >= min_mask_area_ratio:
                    smoke_candidates.append({
                        'area': mask_area,
                        'x': x,
                        'y': y,
                        'img': crop_img,
                        'mask': crop_mask
                    })
                elif mask_area <= 0:
                    non_smoke_positions.append((x, y))

        smoke_candidates.sort(key=lambda x: x['area'], reverse=True)
        selected_smoke = smoke_candidates[:top_smoke_regions]
        smoke_count = len(selected_smoke)
        total_smoke += smoke_count
        for candidate in selected_smoke:
            x, y = candidate['x'], candidate['y']
            cv2.imwrite(os.path.join(output_image_folder, f"{base_name}_{x}_{y}.png"), candidate['img'])
            cv2.imwrite(os.path.join(output_mask_folder, f"mask_{base_name}_{x}_{y}.png"), candidate['mask'] * 255)
            cv2.rectangle(overlay, (x, y),
                          (x + crop_size, y + crop_size),
                          (0, 255, 0), -1)
                # # if np.any(crop_mask):
                # if mask_ratio >= min_mask_area_ratio:
                #     cv2.imwrite(os.path.join(output_image_folder, f"{base_name}_{x}_{y}.png"), crop_img)
                #     cv2.imwrite(os.path.join(output_mask_folder, f"mask_{base_name}_{x}_{y}.png"), crop_mask * 255)
                #     smoke_positions.append((x, y))
                #     smoke_count += 1
                #     total_smoke += 1
                #     cv2.rectangle(overlay, (x, y),
                #               (x + crop_size, y + crop_size),
                #               (0, 255, 0), -1)
                # elif not np.any(crop_mask):
                #     non_smoke_positions.append((x, y))

        # save non-smoke crops
        if output_non_smoke_image_folder and non_smoke_positions:
            # print(f"{img_info['file_name']} has {len(non_smoke_positions)} ")

            # print("num_non_smoke",int(smoke_count * non_smoke_ratio))

            num_non_smoke = min(
                int(smoke_count * non_smoke_ratio),
                max_non_smoke_per_image,
                len(non_smoke_positions)
            )
            # print("num_non_smoke",num_non_smoke)
            if num_non_smoke > 0:
                random.shuffle(non_smoke_positions)
                for x, y in non_smoke_positions[:num_non_smoke]:
                    crop_img = img_cropped[y:y + crop_size, x:x + crop_size]
                    cv2.imwrite(os.path.join(output_non_smoke_image_folder, f"{base_name}_nonsmoke_{x}_{y}.png"),
                                crop_img)
                    total_non_smoke += 1
                    cv2.rectangle(overlay, (x, y),
                                  (x + crop_size, y + crop_size),
                                  (255, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, viz_img, 1 - alpha, 0, viz_img)

        # Add legend
        cv2.putText(viz_img, "Smoke Crops", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz_img, "Non-Smoke Crops", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save visualization
        viz_path = os.path.join(output_image_folder, "..", "visualizations",
                                f"{base_name}_crops.jpg")
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        cv2.imwrite(viz_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 50)
    print(f"SMOKE CROPS: {total_smoke}")
    print(f"NON-SMOKE CROPS: {total_non_smoke}")
    print(f"TOTAL CROPS: {total_smoke + total_non_smoke}")
    print("=" * 50)




def visualize_cropping(image_path,
                       crop_size=512,
                       stride=64):


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

    crop_images_and_masks(
        json_path=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
        image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
        output_image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images"),
        output_mask_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks"),
        output_non_smoke_image_folder=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/non_smoke_images"),
        crop_size=512,
        stride=128,
        non_smoke_ratio=1,
        max_non_smoke_per_image=3
    )

    #
    # sample_image = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/kooks_2__2024-11-15T10-49-07Z_frame_2333_jpg.rf.f7862d504c0ac32843fc5a21a819f14b.jpg")
    # if os.path.exists(sample_image):
    #     visualize_cropping(sample_image)
    # else:
    #     print(f"Sample image not found at {sample_image}")