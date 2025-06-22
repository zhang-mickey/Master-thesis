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
project_root = os.path.abspath(os.path.dirname(__file__) + "/../..")

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

    parser.add_argument("--num_epochs", type=int, default=3, help="epoch number")

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")
    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification_raw.pth"),
                        help="Path to save the trained model")

    # parser.add_argument("--backbone", type=str, default="resnet101",
    #                     help="choose backone")
    # parser.add_argument("--backbone", type=str, default="resnet38d",
    #                     help="choose backone")
    # parser.add_argument("--backbone", type=str, default="transformer",
    #                     help="choose backone")

    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    #                     help="choose backone")

    # parser.add_argument("--CAM_type", type=str, default='TransCAM',
    #                     choices=['grad', 'TransCAM', 'TsCAM'],
    #                     help="CAM type")

    # parser.add_argument("--backbone", type=str, default="conformer",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vgg16d",
    #                     help="choose backone")
    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')

    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    parser.add_argument('--ratio', default=0.1, type=float, help='ratio for ood')
    return parser.parse_args()


if __name__ == "__main__":
    print("Starting training...")
    args = parse_args()

    print(vars(args))
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # %%
    # image_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    image_transform, mask_transform = get_transforms(args.img_size)

    # set random seed
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    if args.use_crop:
        train_dataset = CropDataset(
            # image_dir=args.crop_smoke_image_folder,
            # mask_dir=args.crop_mask_folder,
            # non_smoke_dir=args.crop_non_smoke_folder,
            ijmond_positive_dir=args.Dutch_positive_path,
            ijmond_negative_dir=args.Dutch_negative_path,
            transform=image_transform,
            mask_transform=mask_transform,
            img_size=(args.crop_size, args.crop_size),
            backbone=args.backbone
        )

        total_size = len(train_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_size = total_size

        train_indices = indices[:train_size]

        train_subset = Subset(train_dataset, train_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

        train_dataset_1 = CropDataset(
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

        total_size_1 = len(train_dataset_1)
        indices_1 = list(range(total_size_1))
        random.shuffle(indices_1)
        test_size_1 = total_size_1
        test_indices_1 = indices_1[:test_size_1]
        test_subset_1 = Subset(train_dataset_1, test_indices_1)

        test_loader = DataLoader(test_subset_1, batch_size=args.batch_size, shuffle=False)

        print(f"Training Dataset loaded: {len(train_dataset)} images in total")
        print(f"Test Dataset loaded: {len(train_dataset_1)} images in total")
        print(f"Number of batches: {len(train_loader)}")
        print(f"Number of batches: {len(test_loader)}")

    model = choose_backbone(args.backbone)
    model = model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # warmup_scheduler=LinearLR(optimizer, start_factor=args.warmup_lr/args.lr, total_iters=5)
    # main_scheduler=CosineAnnealingLR(optimizer, T_max=args.num_epochs-5)

    # scheduler = SequentialLR(
    # optimizer,
    # schedulers=[warmup_scheduler, main_scheduler],
    # milestones=[5]  # Switch to main_scheduler after epoch 5
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter = AverageMeter('loss', 'cls_loss', 'f1', 'accuracy', 'loss_entropy', 'bg_loss', 'consistency_loss')

    if args.backbone == 'resnet101' or args.backbone == 'resnet38d' or args.backbone == 'resnet50':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()
            data_load_start = time.time()

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                augmented_images = augment_batch(images)
                mask = mask.to(device)

                optimizer.zero_grad()

                # x,embedded,[f2,f3,f4]
                outputs, _, _ = model(images)
                outputs_aug, _, _ = model(augmented_images)

                outputs = outputs.squeeze(1)
                outputs_aug = outputs_aug.squeeze(1)

                acc = calculate_accuracy(outputs, labels)
                f1 = calculate_f1_score(outputs, labels)
                cls_loss = criterion(outputs, labels)
                consistency_loss = F.mse_loss(outputs, outputs_aug)
                # loss=cls_loss+0.1*consistency_loss
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
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
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

        # Test phase
        print("Starting test phase...")
        test_loss = 0.0
        test_accuracy = 0.0
        test_predictions = []
        test_ground_truth = []

        with torch.no_grad():
            for batch_idx, (images, labels, _, mask) in enumerate(test_loader):
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                outputs = outputs.squeeze(1)

                # Calculate loss and accuracy
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)

                test_loss += loss.item()
                test_accuracy += acc

                # Store predictions and ground truth for metrics calculation
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                test_predictions.extend(predictions.cpu().numpy())
                test_ground_truth.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader)}], ClS Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        # Calculate average metrics
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


    elif args.backbone == 'transformer':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()
            data_load_start = time.time()

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                augmented_images = augment_batch(images)
                mask = mask.to(device)

                optimizer.zero_grad()
                outputs, attns, feature_maps = model(images)
                outputs_aug, attns_aug, features_maps_aug = model(augmented_images)

                outputs = outputs.squeeze(1)
                outputs_aug = outputs_aug.squeeze(1)

                acc = calculate_accuracy(outputs, labels)
                f1 = calculate_f1_score(outputs, labels)
                cls_loss = criterion(outputs, labels)
                consistency_loss = F.mse_loss(outputs, outputs_aug)

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
            f"{args.backbone}_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
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

        # Test phase
        print("Starting test phase...")
        test_loss = 0.0
        test_accuracy = 0.0
        test_predictions = []
        test_ground_truth = []

        with torch.no_grad():
            for batch_idx, (images, labels, _, mask) in enumerate(test_loader):
                images, labels = images.to(device), labels.float().to(device)

                outputs, attns, feature_maps = model(images)

                outputs = outputs.squeeze(1)

                # Calculate loss and accuracy
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)

                test_loss += loss.item()
                test_accuracy += acc

                # Store predictions and ground truth for metrics calculation
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                test_predictions.extend(predictions.cpu().numpy())
                test_ground_truth.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader)}], ClS Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        # Calculate average metrics
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