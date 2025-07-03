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
from post_processing.cam_compare import *
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

    parser.add_argument("--img_size", type=int, default=512, help="the size of image")
    parser.add_argument("--num_class", type=int, default=1, help="the number of classes")

    parser.add_argument("--crop_size", default=512, type=int)

    parser.add_argument("--weights_path", required=False, type=str)

    parser.add_argument("--CAM_type", type=str, default='GradCAM',
                        choices=['grad', 'TransCAM', 'TsCAM'],
                        help="CAM type")

    parser.add_argument("--save_model_path", type=str,
                        default=os.path.join(project_root, "model/model_classification_kd.pth"),
                        help="Path to save the trained model")

    parser.add_argument("--num_epochs", type=int, default=3, help="epoch number")
    # parser.add_argument("--backbone", type=str, default="resnet101",
    #                     help="choose backone")
    # parser.add_argument("--backbone", type=str, default="resnet38d",
    #                     help="choose backone")
    parser.add_argument("--backbone", type=str, default="vit_s",
                        help="choose backone")

    # parser.add_argument("--backbone", type=str, default="resnet50",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="mix_transformer",
    #                     help="choose backone")

    # parser.add_argument("--CAM_type", type=str, default='TransCAM',
    #                     choices=['grad', 'TransCAM', 'TsCAM'],
    #                     help="CAM type")

    # parser.add_argument("--backbone", type=str, default="conformer",
    #                     help="choose backone")

    # parser.add_argument("--backbone", type=str, default="vgg16d",
    #                     help="choose backone")
    parser.add_argument('--kd_alpha', default=0.5, type=float, help='ratio for transfer loss')

    parser.add_argument('--manual_seed', default=42, type=int, help='Manually set random seed')

    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for CAM')
    parser.add_argument('--ratio', default=0.1, type=float, help='ratio for ood')
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

    model_vit = choose_backbone(args.backbone)
    model_resnet50 = choose_backbone("resnet50")

    # model_aug=choose_backbone(args.backbone)
    model_vit = model_vit.to(device)
    model_resnet50 = model_resnet50.to(device)
    # model_aug=model_aug.to(device)
    combined_parameters = list(model_vit.parameters()) + list(model_resnet50.parameters())
    # combined_parameters = list(model.parameters()) + list(model_aug.parameters())
    model_vit.train()
    # model_aug.train()
    model_resnet50.train()

    criterion = nn.BCEWithLogitsLoss()
    criterion_kd = nn.MSELoss()
    optimizer = optim.AdamW(combined_parameters, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    avg_meter = AverageMeter('loss', 'kd_loss', 'cls_loss', 'cls_aug_loss', 'f1', 'accuracy', 'loss_entropy', 'bg_loss',
                             'consistency_loss')

    if args.backbone == 'resnet101' or args.backbone == 'resnet38d' or args.backbone == 'resnet50':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()
            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                augmented_images = augment_batch(images)
                mask = mask.to(device)

                optimizer.zero_grad()

                logits, featmap, _ = model(images)
                logits_aug, featmap_aug, _ = model_aug(augmented_images)

                logits = logits.squeeze(1)
                logits_aug = logits_aug.squeeze(1)

                acc = calculate_accuracy(logits, labels)
                f1 = calculate_f1_score(logits, labels)
                cls_loss = criterion(logits, labels)
                cls_aug_loss = criterion(logits_aug, labels)

                featmap = featmap.clone()
                featmap[featmap < 0] = 0
                max_values = torch.max(torch.max(featmap, dim=3)[0], dim=2)[0]
                featmap = featmap / (max_values.unsqueeze(2).unsqueeze(3) + 1e-8)

                featmap_aug_teacher = featmap_aug.clone().detach()
                featmap_aug_teacher[featmap_aug_teacher < 0] = 0
                max_values = torch.max(torch.max(featmap_aug_teacher, dim=3)[0], dim=2)[0]
                featmap_aug_teacher = featmap_aug_teacher / (max_values.unsqueeze(2).unsqueeze(3) + 1e-8)

                print("featmap", featmap.shape)
                print("featmap_aug_teacher", featmap_aug_teacher.shape)

                kd_loss = criterion_kd(featmap, featmap_aug_teacher)

                loss = cls_aug_loss + kd_loss

                loss.backward()

                optimizer.step()

                avg_meter.add({'loss': loss.item(),
                               'f1': f1,
                               'kd_loss': kd_loss.item(),
                               'cls_loss': cls_loss.item(),
                               'cls_aug_loss': cls_aug_loss.item(),
                               'accuracy': acc})

            avg_loss, kd_loss, cls_aug_loss, cls_loss, avg_acc, avg_f1 = avg_meter.get('loss', 'kd_loss',
                                                                                       'cls_aug_loss', 'cls_loss',
                                                                                       'accuracy', 'f1')
            print(
                f"Epoch [{epoch}/{args.num_epochs}], Loss: {avg_loss:.4f},kd_loss:{kd_loss:.4f},cls_aug_loss:{cls_aug_loss:.4f},cls Loss: {cls_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%,Avg f1:{avg_f1:.4f}")

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


    elif args.backbone == 'vit_b' or args.backbone == 'vit_s':

        cls_loss_history = []
        bg_loss_history = []
        train_accuracies = []
        for epoch in range(1, (args.num_epochs + 1)):

            avg_meter.pop()
            data_load_start = time.time()

            for batch_idx, (images, labels, _, mask) in enumerate(train_loader):
                images, labels = images.to(device), labels.float().to(device)
                # augmented_images = augment_batch(images)
                mask = mask.to(device)

                optimizer.zero_grad()

                logits_vit, attns, featmap_vit = model_vit(images)

                logits_resnet, featmap_resnet, intermediate_feature = model_resnet50(images)

                # featmap_resnet = featmap_resnet.detach()
                # logits_resnet = logits_resnet.detach()

                featmap_resnet = featmap_resnet.view(featmap_resnet.size(0), featmap_resnet.size(1), -1)
                featmap_resnet_inter = intermediate_feature.view(intermediate_feature.size(0),
                                                                 intermediate_feature.size(1), -1)

                # proj_resnet = nn.Conv1d(2048, 128, kernel_size=1).to(device)
                # proj_vit = nn.Conv2d(768, 512, 1).to(device)

                featmap_vit = featmap_vit.view(featmap_vit.size(0), featmap_vit.size(1), -1)
                # print("featmap_vit",featmap_vit.shape)
                # print("featmap_resnet",featmap_resnet.shape)
                # print("featmap_resnet_inter",featmap_resnet_inter.shape)
                # featmap_resnet_inter=proj_resnet(featmap_resnet_inter)
                # featmap_vit=proj_vit(featmap_vit)
                # similarity_matrix = F.cosine_similarity(
                #     featmap_vit.unsqueeze(2),
                #     featmap_resnet.unsqueeze(1),
                #     dim=3
                # )

                # kd_loss = 1 - similarity_matrix.mean()
                # featmap_vit torch.Size([8, 768, 1024])
                # featmap_resnet torch.Size([8, 2, 1024])
                # featmap_resnet_inter torch.Size([8, 2048, 1024])

                kd_loss = compute_similarity(featmap_vit, featmap_resnet, metric='cosine', mode='global',
                                             batch_idx=batch_idx)

                logits_vit = logits_vit.squeeze(1)
                logits_resnet = logits_resnet.squeeze(1)

                acc = calculate_accuracy(logits_vit, labels)
                f1 = calculate_f1_score(logits_vit, labels)
                cls_vit_loss = criterion(logits_vit, labels)
                cls_resnet_loss = criterion(logits_resnet, labels)

                loss = cls_resnet_loss + args.kd_alpha * kd_loss + cls_vit_loss

                loss.backward()
                optimizer.step()
                avg_meter.add({'loss': loss.item(),
                               'kd_loss': kd_loss.item(),
                               'f1': f1,
                               'cls_loss': cls_vit_loss.item(),
                               'cls_aug_loss': cls_resnet_loss.item(),
                               'accuracy': acc})

            avg_loss, kd_loss, cls_vit_loss, cls_resnet_loss, avg_acc, avg_f1 = avg_meter.get('loss', 'kd_loss',
                                                                                              'cls_loss',
                                                                                              'cls_aug_loss',
                                                                                              'accuracy', 'f1')
            print(
                f"Epoch [{epoch}/{args.num_epochs}], Loss: {avg_loss:.4f},kd Loss: {kd_loss:.4f},cls Loss: {cls_vit_loss:.4f},cls_aug_loss:{cls_resnet_loss:.4f}, Avg Accuracy: {avg_acc:.2f}%,Avg f1:{avg_f1:.4f}")

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
        save_resnet_path = os.path.join(
            os.path.dirname(args.save_model_path),
            f"resnet50_{args.CAM_type}_{args.num_epochs}_{os.path.basename(args.save_model_path)}"
        )

        torch.save(model_vit.state_dict(), save_path)
        torch.save(model_resnet50.state_dict(), save_resnet_path)
        print("Training complete! Model saved.")

        model_vit.load_state_dict(torch.load(save_path))

        model_resnet50.load_state_dict(torch.load(save_resnet_path))

        model_vit.eval()
        model_vit.cuda()

        model_resnet50.eval()
        model_resnet50.cuda()

        # Test phase
        print("Starting test phase...")
        test_loss = 0.0
        test_resnet_loss = 0.0
        test_accuracy = 0.0
        test_resnet_accuracy = 0.0
        test_predictions = []
        test_predictions_resnet = []
        test_ground_truth = []

        with torch.no_grad():
            for batch_idx, (images, labels, _, mask) in enumerate(test_loader):
                images, labels = images.to(device), labels.float().to(device)

                outputs, attns, feature_maps = model_vit(images)
                outputs_resnet, attns_resnet, feature_maps_resnet = model_resnet50(images)

                outputs = outputs.squeeze(1)
                outputs_resnet = outputs_resnet.squeeze(1)

                # Calculate loss and accuracy
                loss = criterion(outputs, labels)
                loss_resnet = criterion(outputs_resnet, labels)

                acc = calculate_accuracy(outputs, labels)
                acc_resnet = calculate_accuracy(outputs_resnet, labels)

                test_loss += loss.item()
                test_resnet_loss += loss_resnet.item()
                test_accuracy += acc
                test_resnet_accuracy += acc_resnet

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                predictions_resnet = (torch.sigmoid(outputs_resnet) > 0.5).float()

                test_predictions.extend(predictions.cpu().numpy())
                test_predictions_resnet.extend(predictions_resnet.cpu().numpy())

                test_ground_truth.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"Test Batch [{batch_idx + 1}/{len(test_loader)}], ClS Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)
        avg_test_resnet_loss = test_resnet_loss / len(test_loader)
        avg_test_resnet_accuracy = test_resnet_accuracy / len(test_loader)

        precision = precision_score(test_ground_truth, test_predictions, zero_division=0)
        precision_resnet = precision_score(test_ground_truth, test_predictions_resnet, zero_division=0)

        # recall = recall_score(test_ground_truth, test_predictions, zero_division=0)
        f1 = f1_score(test_ground_truth, test_predictions, zero_division=0)
        f1_resnet = f1_score(test_ground_truth, test_predictions_resnet, zero_division=0)
        # conf_matrix = confusion_matrix(test_ground_truth, test_predictions)
        positive_count = sum(test_ground_truth)
        negative_count = len(test_ground_truth) - positive_count

        print("\n===== Test Results =====")
        print(f"Average CLS ViT Loss: {avg_test_loss:.4f}")
        print(f"Average CLS Resnet Loss: {avg_test_resnet_loss:.4f}")
        print(f"Positive Class Count: {int(positive_count)}")
        print(f"Negative Class Count: {int(negative_count)}")
        print(f"Accuracy: {avg_test_accuracy:.2f}%")
        print(f"Accuracy Resnet: {avg_test_resnet_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Precision Resnet: {precision_resnet:.4f}")
        # print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"F1 Score Resnet: {f1_resnet:.4f}")