import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import os, argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import random_split

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)

from lib.dataset.SmokeDataset import *
from lib.dataset.WeaklyDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.loss.loss import *
from inference.inference import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.dataset.cropDataset import *


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
    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")
    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")
    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root, "result/cam"),
                        help="Path to save the cam")

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
    parser.add_argument("--num_epochs", type=int, default=10, help="epoch number")
    parser.add_argument("--img_size", type=int, default=512, help="the size of image")

    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")
    parser.add_argument("--backbone", type=str, default="transformer",
                        help="choose backone")
    # parser.add_argument("--backbone", type=str, default="deeplabv3plus_resnet101", help="choose backone")

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # infer
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold to pesudo label")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_path = os.path.join(
        os.path.dirname(args.save_model_path),
        f"{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_transform, mask_transform = get_transforms(args.img_size)

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)

    # train_ids, val_ids, test_ids = split_dataset(args.json_path, args.image_folder)

    # print(f"Smoke Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    # non_smoke_train, non_smoke_val, non_smoke_test = split_non_smoke_dataset(args.non_smoke_image_folder)
    # print(f"Non-smoke Dataset split: Train={len(non_smoke_train)}, Val={len(non_smoke_val)}, Test={len(non_smoke_test)}")

    if args.use_crop:
        train_dataset = CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            # args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform
        )
        print(f"Train size: {len(train_dataset)}")

        # Suppose you want 80% training and 20% testing
        total_size = len(train_dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset, test_subset = random_split(train_dataset, [train_size, test_size])
        print(f"Train size: {len(train_subset)} | Test size: {len(test_subset)}")
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = choose_backbone(args.backbone)
    model = model.to(device)

    if args.backbone == "deeplabv3plus_resnet101":
        target_layers = [model.layer4[-1]]  # Last layer of layer4
    elif args.backbone == "transformer":
        target_layers = [model.blocks[-1].norm1]  # Last transformer block
    else:
        target_layers = [list(model.children())[-3]]  # Example fallback
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # generate_cam_for_dataset(
    #     dataloader=train_loader,
    #     model=model,
    #     target_layers=target_layers,
    #     save_dir=args.save_cam_path,
    # )

    # Generate pseudo-labels

    generate_pseudo_labels(
        dataloader=train_loader,
        model=model,
        target_layers=target_layers,
        save_dir=args.save_pseudo_labels_path,
        threshold=0.5
    )