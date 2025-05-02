import torch
import torch.nn as nn
import torch.optim as optim
import json
import cv2
import numpy as np
import sys
import importlib
import os, argparse
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from matplotlib import pyplot as plt
from torch_lr_finder import LRFinder

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

# Add the project root to sys.path
sys.path.append(project_root)
from lib.network.AffinityNet.resnet38_cls import *
from lib.dataset.lr_test import *
from lib.dataset.cropDataset import *
from lib.dataset.aug_cropDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.utils.dark_channel_prior import *
from lib.loss.loss import *
from inference.inference import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.utils.cam import *
from lib.utils.augmentation import *
from lib.utils.image_mask_visualize import *


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised learning")
    # dataset
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

    parser.add_argument("--save_pseudo_labels_path", type=str,
                        default=os.path.join(project_root, "result/pseudo_labels"),
                        help="Path to save the pseudo labels")

    parser.add_argument("--save_cam_path", type=str, default=os.path.join(project_root, "result/cam"),
                        help="Path to save the cam")

    parser.add_argument("--save_visualization_path", type=str,
                        default=os.path.join(project_root, "result/visualization"),
                        help="Path to save the cam")

    parser.add_argument("--smoke5k", type=bool, default=False, help="use smoke5k or not")
    parser.add_argument("--smoke5k_path", type=str, default=os.path.join(project_root, "SMOKE5K/train/"),
                        help="path to smoke5k")

    parser.add_argument("--Rise", type=bool, default=False, help="use Rise non-smoke or not")
    parser.add_argument("--Rise_path", type=str, default=os.path.join(project_root, "Rise/Strong_negative_frames/"),
                        help="path to Rise")

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

    # train
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")

    parser.add_argument("--num_epochs", type=int, default=30, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    # parser.add_argument("--backbone", type=str, default="resnet101",
    #                     help="choose backone")

    parser.add_argument("--backbone", type=str, default="transformer",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    #                     help="choose backone")

    # parser.add_argument("--CAM_type", type=str, default='TransCAM',
    #                     choices=['grad', 'TransCAM', 'TsCAM'],
    #                     help="CAM type")

    # parser.add_argument("--backbone", type=str, default="conformer",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet38d",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vgg16d",
    #                     help="choose backone")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()

    print(vars(args))
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transform, mask_transform = get_transforms(args.img_size)

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if args.use_crop:
        train_dataset = LR_CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size)
        )

        total_size = len(train_dataset)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset, test_subset = random_split(train_dataset, [train_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

        train_dataset_1 = CropDataset(
            args.crop_smoke_image_folder,
            args.crop_mask_folder,
            args.crop_non_smoke_folder,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size)
        )

        total_size = len(train_dataset_1)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size

        # Split dataset
        train_subset_1, test_subset_1 = random_split(train_dataset_1, [train_size, test_size])

        # Create DataLoaders
        train_loader_1 = DataLoader(train_subset_1, batch_size=args.batch_size, shuffle=True)
        test_loader_1 = DataLoader(test_subset_1, batch_size=args.batch_size, shuffle=False)


    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)  # assume returns (logits, attns)
            return outputs[0]  # return only logits


    model = choose_backbone(args.backbone)
    model = ModelWrapper(model)
    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-7)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)

    os.makedirs(os.path.join(project_root, "result/lr_finder"), exist_ok=True)
    save_path = os.path.join(project_root, f"result/lr_finder/lr_finder_{args.backbone}.png")
    lr_finder.plot(skip_start=10, skip_end=5, log_lr=True)
    plt.savefig(save_path)
    print(f"LR Finder plot saved to {save_path}")
    lr_finder.reset()