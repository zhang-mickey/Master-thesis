import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import argparse
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="threshold_experiment")

    parser.add_argument("--threshold", type=float, default=0.3, help="threshold to pesudo label")

    parser.add_argument("--backbone", type=str, default="transformer", help="threshold to pesudo label")

    parser.add_argument("--scale_factor", type=int, default=1, help="scale_factor for crf")

    parser.add_argument("--iteration", type=int, default=5, help="iteration for crf")

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/vit_s_sam_pseudo_labels"),
                        help="Path to save the pseudo labels")
    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")
    parser.add_argument("--sam_model", type=str, default="vit_h",
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help="SAM model type")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="pretrained/sam_vit_h_4b8939.pth",
                        help="SAM checkpoint path")
    return parser.parse_args()


def init_sam_predictor(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,  # how densely points are sampled
        pred_iou_thresh=0.82,  # Threshold for removing low quality or duplicate masks
        stability_score_thresh=0.85,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,  # filter small region
        output_mode="binary_mask"
    )
    return mask_generator


def sam_postprocessing(image, cam, mask_generator, threshold=0.3):
    masks = mask_generator.generate(image)
    cam = (cam[1] > 0.3).astype(np.uint8)

    valid_masks = []
    for mask in masks:
        mask_size_ratio = np.sum(mask['segmentation']) / (cam.shape[0] * cam.shape[1])
        if mask_size_ratio > 0.3:
            continue
        # largest intersection
        intersection = np.sum(cam & mask['segmentation'])
        union = np.sum(cam | mask['segmentation']) + 1e-6
        iou = intersection / union
        # overlap = np.sum(cam * mask['segmentation']) / np.sum(mask['segmentation'])
        if iou > threshold:
            valid_masks.append(mask['segmentation'])

    if valid_masks:
        combined_mask = np.logical_or.reduce(valid_masks)
    else:
        combined_mask = cam.astype(bool)
        # combined_mask = np.zeros_like(cam, dtype=bool)

    return combined_mask.astype(np.uint8)


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-8)
    return iou


if __name__ == "__main__":
    args = parse_args()

    sam_predictor = init_sam_predictor(args)

    image_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/")

    cam_dir = os.path.join(project_root, "final_model/vit_s_GradCAM_0.3_3_pseudo_labels_kd/")
    # cam_dir=os.path.join(project_root, "result/resnet101_GradCAM_0.3_10_pseudo_labels/")
    pseudo_dir = os.path.join(project_root, "final_model/vit_s_GradCAM_0.3_3_pseudo_labels_kd/")
    mask_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/")

    save_pseudo_labels_path = os.path.join(
        os.path.dirname(args.save_pseudo_labels_path),
        f"{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_pseudo_labels_path)}_sam"
    )
    os.makedirs(save_pseudo_labels_path, exist_ok=True)
    print("mask_path", mask_dir)
    print("cam_path", cam_dir)

    samples = []

    for img_name in sorted(os.listdir(image_dir)):
        if img_name.startswith('.'):
            continue

        img_path = os.path.join(image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        cam_path = os.path.join(cam_dir, f"fusion_cam_{base_name}.npy")
        # cam_path = os.path.join(cam_dir, f"fusion_cam_{img_name}")
        # pseudo_path = os.path.join(pseudo_dir, f"fusion_pseudo_label_{img_name}")
        mask_path = os.path.join(mask_dir, f"mask_{img_name}") if mask_dir else None
        if os.path.exists(cam_path) and os.path.exists(mask_path):
            samples.append({
                'image': img_path,
                'cam': cam_path,
                'image_id': base_name,
                # 'pseudo': pseudo_path,
                'mask': mask_path if mask_path and os.path.exists(mask_path) else None,
            })

    optimal_thresholds = []
    fixed_mIOU = []
    # pseudo_mIOU=[]
    sam_fixed_mIOU = []
    # for i in range(15):
    for i in range(len(samples)):
        sample = samples[i]

        img_id = sample['image_id']
        print("image_id", img_id)
        orig_img = cv2.imread(sample['image'])
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        cam = np.load(sample['cam'])
        print(cam.shape)

        gt_mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # Calculate IoU with fixed threshold of 0.3
        fixed_pred = (cam >= args.threshold).astype(np.uint8)
        fixed_iou = compute_iou(fixed_pred, gt_mask)
        fixed_mIOU.append(fixed_iou)

        probs = np.zeros((2, cam.shape[0], cam.shape[1]), dtype=np.float32)

        probs[0] = 1 - cam  # Background probability
        probs[1] = cam  # Foreground probability
        probs[0] = np.power(probs[0], 4)
        probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)

        sam_mask = sam_postprocessing(
            orig_img,
            probs,
            sam_predictor,
            threshold=args.threshold
        )
        sam_iou = compute_iou(sam_mask, gt_mask)
        sam_fixed_mIOU.append(sam_iou)

        if i > 15 and i < 31:
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))

            # Original image
            axes[0].imshow(orig_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Original CAM
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title(f'Original CAM (IoU: {fixed_iou:.3f})')
            axes[1].axis('off')

            # CRF refined CAM
            axes[2].imshow(sam_mask, cmap='jet')
            axes[2].set_title(f'SAM Refined CAM (IoU: {sam_iou:.3f})')
            axes[2].axis('off')

            # Ground truth
            axes[3].imshow(gt_mask, cmap='gray')
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')

            plt.tight_layout()
            os.makedirs("result/visualization/sam_transformer", exist_ok=True)
            plt.savefig(f"result/visualization/sam_transformer/sample_{i}.png")
            plt.close()
        # np.save(
        #                 os.path.join(save_pseudo_labels_path, f"fusion_cam_{img_id}.npy"),
        #                 sam_mask
        #             )

    # fixed_mean_iou=np.mean(fixed_mIOU)
    # pseudp_mean_iou=np.mean(pseudo_mIOU)
    sam_fixed_mean_iou = np.mean(sam_fixed_mIOU)

    # print(f"Mean IoU with (fixed threshold=0.3): {fixed_mean_iou:.4f}")
    # print(f"Mean IoU with (pseudo label): {pseudp_mean_iou:.4f}")
    print(f"Mean IoU with sam (fixed threshold=0.3): {sam_fixed_mean_iou:.4f}")

    # plt.figure(figsize=(12, 6))
    # methods = ['Optimal', 'Fixed=0.3', 'CRF (Optimal)', 'CRF (Fixed=0.3)']
    # ious = [mean_iou, fixed_mean_iou, crf_mean_iou, crf_fixed_mean_iou]
    # colors = ['lightcoral', 'skyblue', 'lightcoral', 'skyblue']

    # bars = plt.bar(methods, ious, color=colors)
    # plt.ylabel('Mean IoU')
    # plt.title('Comparison of Segmentation Methods')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.xticks(rotation=15)

    # # Add value labels on top of each bar
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
    #             f'{height:.4f}', ha='center', va='bottom')

    # plt.tight_layout()
    # plt.savefig("result/visualization/transformer_all_methods_comparison.png")
    # plt.close()