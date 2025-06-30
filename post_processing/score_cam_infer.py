import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
import cv2
import numpy as np
import sys
import os, argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import random_split, Subset
import pydensecrf.densecrf as dcrf
from sklearn.manifold import TSNE
from pytorch_grad_cam import ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.dataset.WeaklyDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from post_processing.single_image_infer import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.dataset.cropDataset import *
from lib.utils.augmentation import *


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    parser.add_argument("--json_path", type=str, default=os.path.join(project_root,
                                                                      "smoke-segmentation.v5i.coco-segmentation/test/_annotations.coco.json"),
                        help="Path to COCO annotations JSON file")
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/test/"),
                        help="Path to the image dataset folder")

    parser.add_argument("--non_smoke_image_folder", type=str, default=os.path.join(project_root, "lib/dataset/frames/"),
                        help="Path to the non-smoke image dataset folder")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

    parser.add_argument("--Dutch", type=bool, default=True, help="use Dutch non-smoke or not")
    parser.add_argument("--Dutch_negative_path", type=str,
                        default=os.path.join(project_root, "frames/manual_negative/"), help="path to Dutch")
    parser.add_argument("--Dutch_positive_path", type=str,
                        default=os.path.join(project_root, "frames/manual_positive/"), help="path to Dutch")

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")

    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root,
                                                                          "result/cam"), help="Path to save the cam")

    parser.add_argument("--crop_smoke_image_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_images/"),
                        help="Path to the cropped smoke image dataset folder")

    parser.add_argument("--crop_mask_folder", type=str,
                        default=os.path.join(project_root, "smoke-segmentation.v5i.coco-segmentation/cropped_masks/"),
                        help="Path to the cropped image dataset mask folder")

    parser.add_argument("--crop_non_smoke_folder", type=str,
                        default=os.path.join(project_root,
                                             "smoke-segmentation.v5i.coco-segmentation/non_smoke_images/"),
                        help="Path to the cropped image dataset mask folder")

    parser.add_argument("--use_crop", type=bool, default=True, help="use cropped image or not")

    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    parser.add_argument("--num_epochs", type=int, default=2, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification_raw.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--backbone", type=str, default="vit_s",
                        help="choose backone")
    # parser.add_argument("--backbone", type=str, default="resnet101",
    # help="choose backone")
    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    # help="choose backone")
    # parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet101", help="choose backone")

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # infer
    parser.add_argument("--threshold", type=float, default=0.3, help="threshold to pesudo label")
    return parser.parse_args()


def generate_pseudo_labels_with_iou(
        model,
        train_loader,
        target_layers,
        save_dir,
        threshold=0.3,
        scales=[0.5, 1.0, 1.5, 2.0]
):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with LayerCAM(model=model, target_layers=target_layers) as cam:

        # 每个 scale 的 IoU 累积字典
        scale_metrics = {s: {'iou_sum': 0.0, 'count': 0} for s in scales}
        total_iou_fused = 0.0
        total_samples = 0

        for idx, (images, labels, _, masks) in enumerate(tqdm(train_loader, desc="Generating Multi-Scale CAM + IoU")):
            images = images.to(device)
            masks = masks.to(device)

            for i in range(images.size(0)):
                img = images[i]
                label = labels[i]
                gt_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                H, W = gt_mask.shape

                cam_accumulator = np.zeros((H, W), dtype=np.float32)

                for scale in scales:
                    h_scaled = int(img.shape[1] * scale)
                    w_scaled = int(img.shape[2] * scale)

                    # Resize image
                    img_scaled = TF.resize(img, [h_scaled, w_scaled])
                    img_scaled = img_scaled.unsqueeze(0)

                    cam_target = [ClassifierOutputTarget(0)]
                    grayscale_cam = cam(input_tensor=img_scaled, targets=cam_target)[0]
                    cam_resized = cv2.resize(grayscale_cam, (W, H))
                    cam_accumulator += cam_resized
                    pseudo_mask_single = (cam_resized > threshold).astype(np.uint8)
                    intersection = np.logical_and(pseudo_mask_single, gt_mask).sum()
                    union = np.logical_or(pseudo_mask_single, gt_mask).sum()
                    iou = intersection / union if union > 0 else 1.0

                    scale_metrics[scale]['iou_sum'] += iou
                    scale_metrics[scale]['count'] += 1

                # Fused CAM
                cam_fused = cam_accumulator / len(scales)
                pseudo_mask_fused = (cam_fused > threshold).astype(np.uint8)

                intersection = np.logical_and(pseudo_mask_fused, gt_mask).sum()
                union = np.logical_or(pseudo_mask_fused, gt_mask).sum()
                iou_fused = intersection / union if union > 0 else 1.0

                total_iou_fused += iou_fused
                total_samples += 1

                save_path = os.path.join(save_dir, f"pseudo_{idx}_{i}.png")
                cv2.imwrite(save_path, pseudo_mask_fused * 255)
    print("\n=== Scale-wise IoU ===")
    for scale in scales:
        count = scale_metrics[scale]['count']
        iou = scale_metrics[scale]['iou_sum'] / count if count > 0 else 0.0
        print(f"Scale {scale:.1f}x → IoU: {iou:.4f}")

    mean_iou_fused = total_iou_fused / total_samples if total_samples > 0 else 0.0
    print(f"\nFused CAM → Mean IoU over {total_samples} samples: {mean_iou_fused:.4f}")
    return mean_iou_fused


if __name__ == "__main__":
    args = parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")
    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform, mask_transform = get_transforms(args.img_size)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

    if args.use_crop:
        train_dataset = CropDataset(
            image_dir=args.crop_smoke_image_folder,
            mask_dir=args.crop_mask_folder,
            # non_smoke_dir=args.crop_non_smoke_folder,
            # ijmond_positive_dir=args.Dutch_positive_path,
            # ijmond_negative_dir=args.Dutch_negative_path,
            test=True,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size),
            backbone=args.backbone
        )
        # print(f"Train size: {len(train_dataset)}")

        total_size = len(train_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_size = int(0.95 * total_size)
        test_size = total_size - train_size

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(train_dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = choose_backbone(args.backbone)
    model = model.to(device)

    if args.backbone == "resnet101":
        target_layers = [model.layer4[-1]]  # Last layer of layer4
    elif args.backbone == "vit_b" or args.backbone == "vit_s":
        target_layers = [model.blocks[-1].norm1]  # Last transformer block
        # target_layers = [
        # # model.blocks[-5].norm1,  #
        # model.blocks[-3].norm1,
        # # model.blocks[-2].norm1, #
        # model.blocks[-2].norm1, ]#
    elif args.backbone == "mix_transformer":
        target_layers = [model.norm4]
        print(target_layers)
    else:
        target_layers = [list(model.children())[-3]]

    model.load_state_dict(torch.load(save_path))
    model.eval()

    # save_cam_path = os.path.join(
    #         os.path.dirname(args.save_cam_path),
    #         f"{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_cam_path)}"
    #     )
    save_pseudo_labels_path = os.path.join(
        os.path.dirname(args.save_pseudo_labels_path),
        f"{args.backbone}_{args.CAM_type}_{args.threshold}_{args.num_epochs}_{os.path.basename(args.save_pseudo_labels_path)}"
    )
    generate_pseudo_labels_with_iou(
        model=model,
        train_loader=train_loader,
        target_layers=target_layers,
        save_dir=save_pseudo_labels_path,
        threshold=0.3,
    )