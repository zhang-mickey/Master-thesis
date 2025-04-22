import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def find_best_threshold_iou(cam, gt_mask):
    # from 0.00 to 1.00

    thresholds = np.linspace(0, 1, 101)
    best_iou = 0
    best_thresh = 0
    for t in thresholds:
        pred_mask = (cam >= t).astype(np.uint8)
        iou = compute_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_thresh = t
    return best_thresh


image_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/")

# cam_path=os.path.join(project_root, "result/transformer_GradCAM_0.3_10_pseudo_labels/")
cam_dir = os.path.join(project_root, "result/resnet101_GradCAM_0.3_10_pseudo_labels/")

mask_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/")

print("mask_path", mask_dir)
print("cam_path", cam_dir)

samples = []

for img_name in sorted(os.listdir(image_dir)):
    if img_name.startswith('.'):
        continue

    img_path = os.path.join(image_dir, img_name)
    cam_path = os.path.join(cam_dir, f"cam_{img_name}")

    mask_path = os.path.join(mask_dir, f"mask_{img_name}") if mask_dir else None
    if os.path.exists(cam_path) and os.path.exists(mask_path):
        samples.append({
            'image': img_path,
            'cam': cam_path,
            'mask': mask_path if mask_path and os.path.exists(mask_path) else None,
        })

optimal_thresholds = []

for i in range(len(samples)):
    sample = samples[i]
    cam = cv2.imread(sample['cam'], cv2.IMREAD_GRAYSCALE) / 255.0  # normalize to [0, 1]

    gt_mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127).astype(np.uint8)  # Ensure it's binary

    best_thresh = find_best_threshold_iou(cam, gt_mask)
    optimal_thresholds.append(best_thresh)

plt.figure(figsize=(8, 5))
sns.histplot(optimal_thresholds, bins=25, kde=True, color="lightcoral")
plt.axvline(np.mean(optimal_thresholds), color='blue', linestyle='--', label='Mean')
plt.axvline(np.median(optimal_thresholds), color='green', linestyle='--', label='Median')

plt.xlabel("Optimal IoU Threshold per Image")
plt.ylabel("Frequency")
plt.title("Distribution of Optimal IoU Thresholds on Training Set")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("result/visualization/optimal_thresholds_distribution.png")
plt.close()