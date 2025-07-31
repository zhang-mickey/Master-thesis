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
from torch.utils.data import random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
from matplotlib import pyplot as plt

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.dirname(__file__) + "/..")

# Add the project root to sys.path
sys.path.append(project_root)
from lib.network.AffinityNet_resnet.resnet38_cls import *
from lib.dataset.SmokeDataset import *
from lib.dataset.WeaklyDataset import *
from lib.dataset.cropDataset import *
from lib.dataset.aug_cropDataset import *
from lib.network.backbone import choose_backbone
from lib.utils.splitdataset import *
from lib.utils.transform import *
from lib.network import *
from lib.utils.dark_channel_prior import *
from lib.loss.loss import *
from lib.utils.metrics import *
from lib.utils.saliencymap import *
from PIL import Image
from lib.utils.pseudo_label import *
from lib.utils.cam import *
from lib.utils.augmentation import *
from lib.utils.image_mask_visualize import *
from train_voc.dataset import *


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

    parser.add_argument("--Dutch", type=bool, default=True, help="use Dutch non-smoke or not")
    parser.add_argument("--Dutch_negative_path", type=str,
                        default=os.path.join(project_root, "frames/manual_negative/"), help="path to Dutch")
    parser.add_argument("--Dutch_positive_path", type=str,
                        default=os.path.join(project_root, "frames/manual_positive/"), help="path to Dutch")

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

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_lr", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=21, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_pcm.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--num_epochs", type=int, default=4, help="epoch number")

    # parser.add_argument("--backbone", type=str, default="resnet50_raw",
    #                     help="choose backone")

    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="choose backone")
    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')

    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    parser.add_argument('--ratio', default=0.1, type=float, help='ratio for ood')

    parser.add_argument("--voc12", type=bool, default=True, help="use VOC12 dataset or not")
    parser.add_argument("--voc12_root", type=str, default=os.path.join(project_root, "VOCdevkit/VOC2012/"),
                        help="path to VOC2012 dataset root directory")
    parser.add_argument("--voc12_list", type=str,
                        default=os.path.join(project_root, "VOCdevkit/VOC2012/ImageSets/Segmentation/"),
                        help="path to VOC2012 dataset list directory")
    parser.add_argument("--voc12_split", type=str, default="train",
                        help="dataset split to use (train, val, test)")

    return parser.parse_args()


if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transform, mask_transform = get_transforms(args.img_size)

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if args.use_crop:
        train_dataset = VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir=args.voc12_list,
            split=args.voc12_split,
            stage='train',
            crop_size=args.crop_size,
            aug=True  # Enable augmentation for training
        )
        # For testing/validation
        test_dataset = VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir=args.voc12_list,
            split='val',  # Use validation split for testing
            stage='val',
            crop_size=args.crop_size,
            aug=False  # No augmentation for testing
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"VOC12 Training Dataset loaded: {len(train_dataset)} images in total")
        print(f"VOC12 Test Dataset loaded: {len(test_dataset)} images in total")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

    model = choose_backbone(args.backbone)
    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter = AverageMeter('loss', 'kd_loss', 'cls_loss', 'cls_aug_loss', 'f1', 'accuracy', 'loss_entropy', 'bg_loss',
                             'consistency_loss')

    if args.backbone == 'resnet50' or args.backbone == 'resnet50_raw':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()

            for batch_idx, (img_name, images, mask, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                mask = mask.to(device)
                optimizer.zero_grad()

                logits, featmap, _ = model(images)

                logits = logits.squeeze(1)

                acc = calculate_accuracy(logits, labels)
                f1 = calculate_f1_score(logits, labels)
                cls_loss = criterion(logits, labels)

                # featmap = featmap.clone()
                # featmap[featmap < 0] = 0
                # max_values = torch.max(torch.max(featmap, dim=3)[0], dim=2)[0]
                # featmap = featmap / (max_values.unsqueeze(2).unsqueeze(3) + 1e-8)

                cls_loss = criterion(logits, labels)
                loss = cls_loss
                loss.backward()

                optimizer.step()

                avg_meter.add({'loss': loss.item(),
                               'f1': f1,
                               'cls_loss': cls_loss.item(),
                               'accuracy': acc})

            avg_loss, cls_loss, avg_acc, avg_f1 = avg_meter.get('loss', 'cls_loss', 'accuracy', 'f1')
            print(
                f"Epoch [{epoch}/{args.num_epochs}], Loss: {avg_loss:.4f},cls Loss: {cls_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%,Avg f1:{avg_f1:.4f}")

            cls_loss_history.append(avg_loss)
            # bg_loss_history.append(avg_bg_loss)
            train_accuracies.append(avg_acc)
            scheduler.step()

        save_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )
        save_loss_path = os.path.join(
            os.path.dirname(args.save_visualization_path),
            f"loss_{args.backbone}_{args.num_epochs}_{os.path.basename(args.save_visualization_path)}"
        )

        torch.save(model.state_dict(), save_path)
        print("Training complete! Model saved.")

        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()

        print("Starting test phase...")
        test_loss = 0.0
        test_accuracy = 0.0
        test_predictions = []
        test_ground_truth = []

        with torch.no_grad():
            for batch_idx, (img_name, images, mask, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)

                test_loss += loss.item()
                test_accuracy += acc
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                test_predictions.extend(predictions.cpu().numpy())
                test_ground_truth.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader)}], ClS Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)

        # precision = precision_score(test_ground_truth, test_predictions, zero_division=0)
        # recall = recall_score(test_ground_truth, test_predictions, zero_division=0)
        f1 = f1_score(test_ground_truth, test_predictions, zero_division=0)
        # conf_matrix = confusion_matrix(test_ground_truth, test_predictions)
        positive_count = sum(test_ground_truth)
        negative_count = len(test_ground_truth) - positive_count

        print("\n===== Test Results =====")
        print(f"Average CLS Loss: {avg_test_loss:.4f}")
        print(f"Positive Class Count: {int(positive_count)}")
        print(f"Negative Class Count: {int(negative_count)}")
        print(f"Accuracy: {avg_test_accuracy:.2f}%")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")