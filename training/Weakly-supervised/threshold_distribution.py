import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="threshold_experiment")

    parser.add_argument("--threshold", type=float, default=0.3, help="threshold to pesudo label")

    parser.add_argument("--model_cam", type=str, default="transformer", help="threshold to pesudo label")

    parser.add_argument("--scale_factor", type=int, default=1, help="scale_factor for crf")

    parser.add_argument("--iteration", type=int, default=10, help="iteration for crf")

    return parser.parse_args()


def crf_inference(img, probs, t=10, scale_factor=1, labels=1):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    # If the initial CAM is not strong or confident (low activation),
    # the CRF can easily suppress it further instead of refining it.
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    print(probs.shape)  # Should be (n_labels, H, W)
    print(probs.max(), probs.min())  # Make sure it's a softmax output
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    # CRF is using the original image for pairwise potential computation.
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=10, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


# def crf_with_alpha():


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-8)
    return iou


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
    return best_thresh, best_iou


if __name__ == "__main__":
    args = parse_args()

    image_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/")

    cam_dir = os.path.join(project_root, "final_model/transformer_GradCAM_0.3_10_pseudo_labels/")
    # cam_dir=os.path.join(project_root, "result/resnet101_GradCAM_0.3_10_pseudo_labels/")
    pseudo_dir = os.path.join(project_root, "final_model/transformer_GradCAM_0.3_10_pseudo_labels/")
    mask_dir = os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/")

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
                # 'pseudo': pseudo_path,
                'mask': mask_path if mask_path and os.path.exists(mask_path) else None,
            })

    optimal_thresholds = []
    mIOU = []
    fixed_mIOU = []
    # pseudo_mIOU=[]
    crf_mIOU = []
    crf_fixed_mIOU = []
    for i in range(len(samples)):
        sample = samples[i]
        orig_img = cv2.imread(sample['image'])
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        cam = np.load(sample['cam'])
        # cam = cv2.imread(sample['cam'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # normalize to [0, 1]
        # pseudo_label=cv2.imread(sample['pseudo'], cv2.IMREAD_GRAYSCALE) / 255.0  # normalize to [0, 1]

        gt_mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Ensure it's binary
        # Calculate IoU with optimal threshold
        best_thresh, best_iou = find_best_threshold_iou(cam, gt_mask)
        mIOU.append(best_iou)
        optimal_thresholds.append(best_thresh)

        # Calculate IoU with fixed threshold of 0.3
        fixed_pred = (cam >= args.threshold).astype(np.uint8)
        fixed_iou = compute_iou(fixed_pred, gt_mask)
        fixed_mIOU.append(fixed_iou)

        # pseudo_iou=compute_iou(pseudo_label, gt_mask)
        # pseudo_mIOU.append(pseudo_iou)

        probs = np.zeros((2, cam.shape[0], cam.shape[1]), dtype=np.float32)

        probs[0] = 1 - cam  # Background probability
        probs[1] = cam  # Foreground probability
        probs = np.power(probs, 0.5)
        probs = probs / (probs.sum(axis=0, keepdims=True) + 1e-8)
        # Run CRF inference
        refined_probs = crf_inference(orig_img, probs, t=args.iteration, scale_factor=1, labels=2)

        refined_cam = refined_probs[1]  # Get foreground probability

        # Find best threshold with CRF
        crf_best_thresh, crf_best_iou = find_best_threshold_iou(refined_cam, gt_mask)
        crf_mIOU.append(crf_best_iou)

        crf_fixed_pred = (refined_cam >= args.threshold).astype(np.uint8)
        crf_fixed_iou = compute_iou(crf_fixed_pred, gt_mask)
        crf_fixed_mIOU.append(crf_fixed_iou)

        if i < 6:  # Only visualize first 5 samples to avoid too many plots
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))

            # Original image
            axes[0].imshow(orig_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Original CAM
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title(f'Original CAM (IoU: {best_iou:.3f})')
            axes[1].axis('off')

            # CRF refined CAM
            axes[2].imshow(refined_cam, cmap='jet')
            axes[2].set_title(f'CRF Refined CAM (IoU: {crf_best_iou:.3f})')
            axes[2].axis('off')

            # Ground truth
            axes[3].imshow(gt_mask, cmap='gray')
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')

            plt.tight_layout()
            os.makedirs("result/visualization/crf_transformer", exist_ok=True)
            plt.savefig(f"result/visualization/crf_transformer/sample_{i}.png")
            plt.close()

    mean_iou = np.mean(mIOU)
    fixed_mean_iou = np.mean(fixed_mIOU)
    # pseudp_mean_iou=np.mean(pseudo_mIOU)
    crf_mean_iou = np.mean(crf_mIOU)
    crf_fixed_mean_iou = np.mean(crf_fixed_mIOU)

    print(f"Mean IoU with (optimal threshold): {mean_iou:.4f}")
    print(f"Mean IoU with (fixed threshold=0.3): {fixed_mean_iou:.4f}")
    # print(f"Mean IoU with (pseudo label): {pseudp_mean_iou:.4f}")
    print(f"Mean IoU with CRF (optimal threshold): {crf_mean_iou:.4f}")
    print(f"Mean IoU with CRF (fixed threshold=0.3): {crf_fixed_mean_iou:.4f}")
    print(f"Mean Threshold: {np.mean(optimal_thresholds):.4f}")
    print(f"Median Threshold: {np.median(optimal_thresholds):.4f}")

    # resnet101
    # Mean IoU: 0.2601
    # Mean Threshold: 0.4117
    # Median Threshold: 0.4000

    plt.figure(figsize=(12, 6))
    methods = ['Optimal', 'Fixed=0.3', 'CRF (Optimal)', 'CRF (Fixed=0.3)']
    ious = [mean_iou, fixed_mean_iou, crf_mean_iou, crf_fixed_mean_iou]
    colors = ['lightcoral', 'skyblue', 'lightcoral', 'skyblue']

    bars = plt.bar(methods, ious, color=colors)
    plt.ylabel('Mean IoU')
    plt.title('Comparison of Segmentation Methods')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("result/visualization/transformer_all_methods_comparison.png")
    plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.bar(['Without CRF', 'With CRF'], [mean_iou, crf_mean_iou], color=['lightcoral', 'lightblue'])
    # plt.ylabel('Mean IoU')
    # plt.title('Effect of CRF Refinement on CAM Segmentation')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig("result/visualization/transformer_crf_comparison.png")
    # plt.close()

    # plt.figure(figsize=(8, 5))
    # sns.histplot(optimal_thresholds, bins=25, kde=True, color="lightcoral")
    # plt.axvline(np.mean(optimal_thresholds), color='blue', linestyle='--', label='Mean')
    # plt.axvline(np.median(optimal_thresholds), color='green', linestyle='--', label='Median')

    # plt.xlabel("Optimal IoU Threshold per Image")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Optimal IoU Thresholds on Training Set")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # plt.savefig("result/visualization/transformer_optimal_thresholds_distribution.png")
    # plt.close()  